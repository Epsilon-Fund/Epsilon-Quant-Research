"""Stage B — transparent fair-value model (features -> probability + band).

No LLM tokens are spent here. The model is a decayed log-odds evidence update:

    FV_t = sigmoid( logit(p0) + alpha * A_t )
    A_t  = lambda^(days elapsed) * A_{t-1} + S_t          (evidence state, clipped)
    S_t  = sum over NEW articles of c_i,   clipped         (daily evidence score)
    c_i  = relevance * phase_weight * strength * direction * novelty_factor * w_src

where p0 is the market's onboarding prior (five-perspective LLM ensemble, set once
at onboarding and re-anchored only with a documented trigger), and the per-article
features come from the Stage A cache (features.py). Every term is inspectable: the
dashboard's FV-construction breakdown lists each article's marginal pp effect and
they sum exactly to FV - p0.

Calibration: alpha (the single fitted parameter) is fit on RESOLVED OUTCOMES — the
v0 gate archive (10 resolved markets, ~56 lookahead-free packets) at build time,
refit as the forward ledger settles. lambda / floors / phase weights are declared
constants (n is far too small to fit them honestly); see fv_params.json.

Market types (config.LIVE_MARKETS[slug]["mtype"]):
  slow  — structural questions (elections, scheduled decisions): slower evidence
          decay, tighter band floor.
  shock — event-driven geopolitics: faster decay, wider band floor. We hold no
          insider information, so shock misses are expected and honestly scored.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .config import DATA

PARAMS_PATH = Path(__file__).with_name("fv_params.json")
STATE_PATH = DATA / "fv_state.json"
PRIORS_PATH = DATA / "priors.json"

PHASE_W = {"none": 0.0, "speculative": 0.25, "planned": 0.5,
           "in_progress": 0.8, "completed": 1.0}
DIR = {"toward_yes": 1.0, "toward_no": -1.0, "neutral": 0.0}
# Scheme B: flat source weights (curated whitelist). Scheme A (RSP tiers + Iffy
# blocklist) plugs in here once signed off.
SOURCE_W_DEFAULT = 1.0
S_CLIP = 2.5           # max |daily evidence score|
A_CLIP = 6.0           # max |cumulative evidence state|
BAND_CAP_PP = 35.0     # matches the v0 gate's p90-sanity bar
RELEVANT_MIN = 0.4     # article counts as "relevant" for band/flag purposes

DEFAULT_PARAMS = {
    "alpha": 0.35,             # placeholder until fit_alpha() runs; overwritten by fit
    # GDELT attention-burst amplification (v2.1): S'_t = S_t * (1 + gamma*max(0, vol_z)).
    # Direction still comes ONLY from Stage-A article evidence; a world-news volume
    # burst can amplify a directional day, never create one (S_t=0 stays 0).
    # DECLARED at 1.0 (a 1-sigma attention burst doubles the day's evidence), NOT
    # fitted: the in-sample Brier on the 53-pair archive improves monotonically in
    # gamma with no plateau — burst-saturation overfit, not a measurable elasticity.
    # 0 = feature off (also the graceful value when the BigQuery series is absent).
    "gamma": 1.0,
    # Per-day evidence decay — DECLARED (n too small to fit): shock news goes stale
    # fast; structural questions retain evidence longer. The archive comparison
    # preferred faster decay; these are round mid-range judgment values.
    "lambda": {"slow": 0.9, "shock": 0.6},
    "floor_pp": {"slow": 8.0, "shock": 12.0},  # band half-width floors (declared)
    # Slow/structural guards (DECLARED): a drip of mildly-positive daily coverage is
    # NOT mounting evidence — a slow market's FV may move only on decisive days
    # (|S_t| >= s_min), and its total news-driven shift from the prior is capped in
    # LOGIT units (alpha-invariant: an earlier A-unit cap silently loosened when
    # alpha was refit — the Dems-House 98.9%/92.2% worked examples). null = no cap
    # (shock markets: a completed event SHOULD saturate the sigmoid).
    "s_min": {"slow": 0.5, "shock": 0.0},
    "shift_clip_logits": {"slow": 1.5, "shock": None},
    "fitted_on": None, "n_pairs": 0, "n_markets": 0,
    "notes": "defaults — not yet fit on resolved outcomes",
}
# Decisive-signal weighting (design choice, informed by the archive comparison where
# dominance-style aggregation beat plain sums at every decay tested): the strongest
# new signal each way counts fully; everything else corroborates at this weight.
CORROBORATION_W = 0.25


def load_params() -> dict:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text())
    return dict(DEFAULT_PARAMS)


def logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def contribution(f: dict, source_w: float = SOURCE_W_DEFAULT) -> float:
    """Signed evidence contribution of one article's features. 0 for irrelevant/neutral."""
    if f is None:
        return 0.0
    novelty_factor = 0.5 + 0.5 * f.get("novelty", 0.5)
    return (f["relevance"] * PHASE_W[f["event_phase"]] * f["strength"]
            * DIR[f["stance"]] * novelty_factor * source_w)


def effective_weights(cs: list[float]) -> list[float]:
    """Decisive-signal weights: the single strongest positive and strongest negative
    contribution get weight 1.0; every other non-zero contribution corroborates at
    CORROBORATION_W. Markets react to the most decisive new signal; a pile of weak
    background items must not drown one confirmed development (the v0 archive's
    Iran-deal packets are the worked example)."""
    pos = [c for c in cs if c > 0]
    neg = [c for c in cs if c < 0]
    i_pos = cs.index(max(pos)) if pos else -1
    i_neg = cs.index(min(neg)) if neg else -1
    return [1.0 if i in (i_pos, i_neg) else (CORROBORATION_W if cs[i] != 0 else 0.0)
            for i in range(len(cs))]


def daily_score(feats: list[dict]) -> float:
    """S_t over the day's NEW articles with decisive-signal weighting."""
    cs = [contribution(r.get("features")) for r in feats]
    if not cs:
        return 0.0
    ws = effective_weights(cs)
    s = sum(w * c for w, c in zip(ws, cs))
    return max(-S_CLIP, min(S_CLIP, s))


def amplify(s_t: float, vol_z: float | None, gamma: float) -> float:
    """GDELT burst amplification of a day's directional evidence (S-clip preserved).

    vol_z None (no series / short baseline) or gamma 0 -> unchanged. Only positive
    bursts amplify; quiet days never dampen Stage-A evidence."""
    if vol_z is None or gamma <= 0.0 or s_t == 0.0:
        return s_t
    s = s_t * (1.0 + gamma * max(0.0, vol_z))
    return max(-S_CLIP, min(S_CLIP, s))


def step_state(prev: dict | None, date: str, s_t: float, lam: float,
               a_clip: float = A_CLIP, s_min: float = 0.0) -> dict:
    """Advance the evidence state one observation day (decay by elapsed days, add S_t).

    s_min: significance threshold — days with |S_t| below it contribute nothing
    (slow-market drip filter). a_clip: per-type evidence accumulation cap."""
    if abs(s_t) < s_min:
        s_t = 0.0
    if prev is None:
        a = s_t
    else:
        # same-date re-run is a no-op step (lam^0 = 1, no new articles -> s_t = 0),
        # so re-publishing a day never double-decays the state
        d_days = max(0, (datetime.fromisoformat(date) -
                         datetime.fromisoformat(prev["date"])).days)
        a = (lam ** d_days) * prev["A"] + s_t
    return {"date": date, "A": max(-a_clip, min(a_clip, a))}


def type_params(params: dict, mtype: str) -> dict:
    """Resolve per-market-type knobs with backward-compatible defaults."""
    return {"lam": params["lambda"].get(mtype, params["lambda"]["shock"]),
            "a_clip": A_CLIP,
            "s_min": params.get("s_min", {}).get(mtype, 0.0),
            "shift_clip": params.get("shift_clip_logits", {}).get(mtype)}


def fair_value(p0_pct: float, a_state: float, alpha: float,
               shift_clip: float | None = None) -> float:
    """FV in percent, clipped to [1, 99]. shift_clip caps the total news-driven
    logit shift from the prior (slow-market guard, alpha-invariant)."""
    shift = alpha * a_state
    if shift_clip is not None:
        shift = max(-shift_clip, min(shift_clip, shift))
    fv = sigmoid(logit(p0_pct / 100.0) + shift) * 100.0
    return min(99.0, max(1.0, fv))


def band_half_pp(feats_72h: list[dict], mtype: str, params: dict) -> float:
    """Band half-width in pp: type floor + evidence-dispersion term − volume shrink.

    dispersion = population std of signed unit contributions (phase*strength*dir)
    across relevant articles; <2 relevant articles = uninformed default 0.5."""
    floor = params["floor_pp"].get(mtype, params["floor_pp"]["shock"])
    units = [PHASE_W[f["features"]["event_phase"]] * f["features"]["strength"]
             * DIR[f["features"]["stance"]]
             for f in feats_72h
             if f.get("features") and f["features"]["relevance"] >= RELEVANT_MIN]
    if len(units) < 2:
        disp = 0.5
    else:
        mean = sum(units) / len(units)
        disp = math.sqrt(sum((u - mean) ** 2 for u in units) / len(units))
    half = floor + 18.0 * disp - 1.5 * min(len(units), 6)
    return round(min(BAND_CAP_PP, max(floor, half)), 1)


def n_relevant(feats_72h: list[dict]) -> int:
    return sum(1 for f in feats_72h
               if f.get("features") and f["features"]["relevance"] >= RELEVANT_MIN)


def divergence_flag(fv_pct: float, mid_pct: float, half_pp: float, n_rel: int,
                    gap_min_pp: float = 15.0, half_max_pp: float = 12.0,
                    n_rel_min: int = 5) -> dict:
    """Public divergence flag: |FV − mid| >= 15pp AND confidence (band half-width
    <= 12pp AND >= 5 relevant articles in 72h). Rationale: 15pp ≈ 2x the v0 median
    |gap| (7.5pp) so only tail disagreements flag; the confidence leg means a flag
    can never fire on a wide band or thin evidence. Shock-type markets (floor 12pp)
    flag only at maximum confidence by construction. Informational only — flags are
    'where our model most disagrees', never an edge claim."""
    gap = fv_pct - mid_pct
    return {"gap_pp": round(gap, 1),
            "diverges": abs(gap) >= gap_min_pp,
            "confident": half_pp <= half_max_pp and n_rel >= n_rel_min,
            "flag": abs(gap) >= gap_min_pp and half_pp <= half_max_pp and n_rel >= n_rel_min,
            "rule": f"|gap|>={gap_min_pp:g}pp AND band half<={half_max_pp:g}pp "
                    f"AND >= {n_rel_min} relevant articles/72h"}


def breakdown(p0_pct: float, prev_state: dict | None, date: str, day_feats: list[dict],
              mtype: str, params: dict, vol_z: float | None = None) -> dict:
    """FV construction, article by article. Marginal pp effects sum exactly to
    FV − p0 (sequential marginals over: decay carry, then articles by |c| desc).
    vol_z folds the GDELT burst amplification into the per-article scale so the
    displayed steps still sum to the published FV."""
    alpha = params["alpha"]
    tp = type_params(params, mtype)
    lam, a_clip = tp["lam"], tp["a_clip"]
    base_logit = logit(p0_pct / 100.0)

    carry_a = 0.0
    if prev_state is not None:
        d_days = max(0, (datetime.fromisoformat(date) -
                         datetime.fromisoformat(prev_state["date"])).days)
        carry_a = (lam ** d_days) * prev_state["A"]

    cs = [contribution(r.get("features")) for r in day_feats]
    ws = effective_weights(cs) if cs else []
    rows = []
    for r, c, w in zip(day_feats, cs, ws):
        if abs(w * c) < 1e-9:
            continue
        rows.append({"title": r["article"].get("title", "")[:140],
                     "domain": r["article"].get("domain", ""),
                     "c": round(w * c, 4), "weight": w, "features": r.get("features")})
    rows.sort(key=lambda x: -abs(x["c"]))

    # mirror the daily pipeline exactly: S-clip pro-rata, then burst amplification,
    # then the slow-market drip filter — allocated back onto articles via `scale`
    s_raw = sum(x["c"] for x in rows)
    scale = 1.0 if abs(s_raw) <= S_CLIP or s_raw == 0 else S_CLIP / abs(s_raw)
    s_clipped = max(-S_CLIP, min(S_CLIP, s_raw))
    s_eff = amplify(s_clipped, vol_z, params.get("gamma", 0.0))
    if s_clipped != 0:
        scale *= s_eff / s_clipped
    below_threshold = abs(s_eff) < tp["s_min"]
    if below_threshold:
        scale = 0.0   # day not decisive enough for this market type: zero effect

    def _fv_at(a: float) -> float:
        return fair_value(p0_pct, min(a_clip, max(-a_clip, a)), alpha, tp["shift_clip"])

    running_a = carry_a
    prev_fv = _fv_at(running_a)
    carry_pp = prev_fv - p0_pct
    steps = []
    for x in rows:
        running_a += x["c"] * scale
        fv_here = _fv_at(running_a)
        steps.append({**{k: x[k] for k in ("title", "domain", "c")},
                      "pp_effect": round(fv_here - prev_fv, 2)})
        prev_fv = fv_here
    return {"p0_pct": p0_pct, "carry_pp": round(carry_pp, 2), "alpha": alpha,
            "lambda": lam, "clip_scale": round(scale, 3),
            "drip_filtered": below_threshold, "articles": steps,
            "fv_pct": round(prev_fv, 1)}


# ---------------------------------------------------------------- calibration --

def fit_alpha(pairs: list[dict], grid_max: float = 6.0, step: float = 0.05) -> dict:
    """Fit the single evidence weight on resolved outcomes.

    pairs: [{p0_pct, A, y, family, shift_clip?}] — one per (market, snapshot).
    Returns the grid-search result with pooled Briers; per-family honesty is the
    caller's job. shift_clip is honored per pair so slow-market guards hold at
    every alpha candidate (alpha-invariant by construction)."""
    def brier(alpha: float) -> float:
        return sum((fair_value(x["p0_pct"], x["A"], alpha,
                               x.get("shift_clip")) / 100.0 - x["y"]) ** 2
                   for x in pairs) / len(pairs)

    grid = [round(i * step, 2) for i in range(int(grid_max / step) + 1)]
    scores = {a: brier(a) for a in grid}
    best = min(scores, key=scores.get)
    return {"alpha": best, "brier_at_best": round(scores[best], 4),
            "brier_prior_only": round(scores[0.0], 4),
            "n_pairs": len(pairs),
            "n_markets": len({x.get("slug") for x in pairs}),
            "curve": [{"alpha": a, "brier": round(b, 4)} for a, b in scores.items()]}


def save_params(params: dict) -> None:
    PARAMS_PATH.write_text(json.dumps(params, indent=1))


# ---------------------------------------------------------------- state I/O ----

def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1))


def load_priors() -> dict:
    return json.loads(PRIORS_PATH.read_text()) if PRIORS_PATH.exists() else {}


def save_prior(slug: str, p0_pct: float, method: str, rationale: str,
               estimates_pct: list[float] | None = None) -> None:
    PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    priors = load_priors()
    priors[slug] = {"p0_pct": round(p0_pct, 1), "set_on": datetime.now().strftime("%Y-%m-%d"),
                    "method": method, "rationale": rationale[:400],
                    "estimates_pct": estimates_pct or []}
    PRIORS_PATH.write_text(json.dumps(priors, indent=1))
