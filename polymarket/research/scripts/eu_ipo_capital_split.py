"""EU retail IPO subscription — capital-split recommender + rolling fill-rate track record.

PLAIN ENGLISH
-------------
You want to subscribe to a retail IPO (e.g. SpaceX / SPCX) across the European brokers you
can actually use -- Revolut, Trade Republic, DEGIRO, Interactive Brokers. This tool turns a
total capital budget `C` and an execution-risk `tilt` into a recommended euro split across
those brokers, respecting each broker's min/max, and explains the reasoning. It also reads
your post-deal subscription logs and computes a rolling per-broker realised fill rate, which
feeds back into future recommendations.

THE CORE IDEA (this is the whole point -- read this first)
----------------------------------------------------------
With flat pro-rata, your shares at a broker = fill_rate x your subscription. So the ONLY thing
that decides where a euro buys you more shares is the broker's FILL RATE. If Revolut fills 25%
and Trade Republic fills 10%, then EUR 1,000 at Revolut = 250 shares but EUR 1,000 at TR = 100
shares -- same money, 2.5x the shares. We want to ESTIMATE each broker's fill rate and put more
money where it is higher. That is the entire job; everything else is bookkeeping.

  shares_b = fill_rate_b x subscription_b        <- maximise by sending money to high fill_rate
  split:  weight_b  ∝  fill_rate_b ** tilt        (tilt = how hard to concentrate; see below)

We estimate fill_rate_b in a strict preference order, using the best available:
  1. REALISED fill rate from your own logs (the mean of past realised fill fractions). The real thing.
  2. A RESEARCH-BASED PRIOR -- a deal-level estimate from public evidence (e.g. SpaceX ~5% retail
     fill). When the evidence shows no per-broker difference, this prior is EQUAL across brokers,
     so the fill-driven split is 50/50 (you cannot honestly steer without per-broker evidence).
  3. The MATURITY PRIOR (gut trust) -- only if there is no fill estimate at all. USER-EDITABLE.
So on a brand-new deal with no logs, the split rests on the research prior (equal => 50/50), NOT on
the gut maturity number; the maturity tilt only bites if you clear the research prior too.

THE TILT KNOB (how hard to chase the higher fill rate)
------------------------------------------------------
    tilt = 0  -> ignore the score, split EQUALLY (diversify; you have no reason to prefer one).
    tilt = 1  -> split PROPORTIONAL to fill rate (double the fill rate => double the money).
    tilt > 1  -> concentrate harder on the higher-fill broker (tilt -> inf = all at the best).

WHY NOT JUST PUT 100% AT THE HIGHEST-FILL BROKER?
-------------------------------------------------
Two honest reasons, both small caveats, not the main event:
  1. The fill-rate estimate is NOISY -- a couple of deals is a tiny sample, so the "best" broker
     today may not be best next time.
  2. Each broker is a SINGLE SEALED draw: you cannot see sub-allocations or demand, so phi_b is
     random and unknown in advance. Going 100% bets everything on one random allocation.
Spreading some money hedges both. Formally, with independent phi_b the variance of total fill is
Var[F] = B^2 * sum_b w_b^2 * sigma_b^2 (no covariance terms), which equal weights minimise -- but
that is the tie-breaker, NOT the objective. The objective is: go where you get filled more.

DEFINITIONS USED BELOW
----------------------
- WITHIN a broker: flat pro-rata -> everyone gets the same phi_b; your size scales absolute
  shares, not your percentage.
- ACROSS brokers: independent sealed sub-allocations -> phi_b are independent and unobservable.
- weight w_b = fraction of the subscription budget at broker b; subscription S_b = w_b * B,
  B = oversubscribe * C; total filled F = sum_b phi_b * S_b, so E[F] = B * sum_b w_b * mu_b.

THE OVER-SUBSCRIPTION WARNING (built into the verdict)
------------------------------------------------------
With a flat pro-rata fill, a low phi_b means you receive few shares, so it is tempting to
SUBSCRIBE FOR MORE than you actually want, to scale the absolute fill back up. The trap:
phi_b is unobservable and can come in HIGH (even 1.0). If you over-subscribe (total
subscription > C) and the fill lands full, you are FULLY ALLOCATED at the inflated size --
you must fund and hold far more exposure than you intended, and you blocked more cash than
planned. The safe default is `oversubscribe = 1.0`: total subscription == C, so the worst
case (phi = 1 everywhere) buys exactly your budget. Any oversubscribe > 1.0 is flagged with
the worst-case funded amount = oversubscribe * C.

SOURCES / VAULT ANCHOR
----------------------
Only Trade Republic is documented in the vault: in-app IPO subscriptions for European retail,
allocation at the official allocation price, PRO-RATA allocation if demand is high, and a
1 EUR settlement fee (see notes/overview/market_maps/spacex_ipo_coworker_addendum.md). Revolut,
DEGIRO and Interactive Brokers carry NO vault facts -- every field for them is UNKNOWN and must
be filled in by hand before trusting a euro number. This tool never invents broker facts;
UNKNOWN stays UNKNOWN and only affects the split through the user-supplied maturity prior.

Companion notes:
  notes/overview/market_maps/eu_ipo_broker_subscription_model.md  (the decision model)
  notes/overview/market_maps/IPO Subscriptions/_template.md            (per-deal log template)
  notes/overview/market_maps/IPO Subscriptions/SpaceX_SPCX_2026-06.md  (instantiated SpaceX deal)

RUN
---
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital 10000 --tilt 1.0
    PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital 10000 --json
    PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --track-record
    PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --selftest
(stdlib-only; plain `python3 scripts/eu_ipo_capital_split.py ...` also works.)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

INF = float("inf")


# --------------------------------------------------------------------------------------
# Broker model
# --------------------------------------------------------------------------------------
@dataclass
class Broker:
    """One candidate broker. `None` on a fact field means UNKNOWN (not in the vault).

    maturity is a USER-PROVIDED execution-trust prior in (0, 1]; it is the ONLY field that
    is allowed to be a subjective estimate, and it is the only thing the `tilt` acts on.
    """

    name: str
    maturity: float                       # user prior, (0, 1]; higher = more proven/trusted
    eligible: Optional[bool] = None       # eligible in Justin's country/account? UNKNOWN
    offers_subscription: Optional[bool] = None  # offers retail IPO subscription for THIS deal?
    fee_fixed: Optional[float] = None     # fixed settlement/commission fee (currency units)
    fee_pct: Optional[float] = None       # percentage fee on notional
    fee_ccy: Optional[str] = None         # currency of the fee
    min_subscription: Optional[float] = None  # broker min subscription (currency)
    max_subscription: Optional[float] = None  # broker max subscription (currency)
    fx_cash_block: Optional[str] = None   # how cash is blocked / FX handled
    allocation_method: Optional[str] = None   # e.g. "flat pro-rata"
    realized_fill_rate: Optional[float] = None   # rolling mean phi_b from track record (best)
    realized_fill_sd: Optional[float] = None      # rolling sd of phi_b
    prior_fill_rate: Optional[float] = None       # research-based DEAL-LEVEL fill estimate (2nd best)
    notes: str = ""


def default_brokers() -> list[Broker]:
    """The candidate brokers Justin actually uses, seeded with vault + user-provided facts.

    SCOPE (user, 2026-06-09): DEGIRO and Interactive Brokers are OUT OF SCOPE ("idc about"),
    so the candidate set is Revolut + Trade Republic only.

    FACTS:
      - Revolut: user-provided (2026-06-09) -- offers the SPCX subscription, NO fee, $500 min.
      - Trade Republic: vault (spacex_ipo_coworker_addendum.md) -- pro-rata, official price, 1 EUR fee.
    The `maturity` numbers are USER PRIORS, not facts. UNKNOWN = not provided; confirm by hand.
    """
    return [
        Broker(
            name="Revolut",
            maturity=0.90,  # USER PRIOR: user weights Revolut highest (most proven to them)
            eligible=True,  # USER (2026-06-09): Justin can use Revolut for this subscription
            offers_subscription=True,  # USER (2026-06-09): Revolut DOES offer the SPCX subscription
            fee_fixed=0.0,  # USER (2026-06-09): no fee
            fee_ccy="USD",
            min_subscription=500.0,  # USER (2026-06-09): $500 minimum (USD-denominated)
            max_subscription=None,   # USER (2026-06-09): no maximum
            allocation_method=None,  # UNKNOWN (model assumes flat pro-rata for all brokers)
            # RESEARCH PRIOR (2026-06-09, SpaceX deal-level): no per-broker fill evidence exists,
            # so the same ~5% deal-level estimate is used for both brokers (=> equal split on fill
            # grounds). Reset/clear per deal; replace with realised fill once SpaceX allocates.
            prior_fill_rate=0.05,
            fx_cash_block="$500 min is USD-denominated; reconcile FX vs a EUR budget yourself",
            notes="USER FACTS (2026-06-09): offers SPCX subscription, no fee, $500 min (USD), NO max, "
            "NO day-1 sell limit. UNKNOWN: confirmed allocation method. Mind the FX vs TR's EUR.",
        ),
        Broker(
            name="Trade Republic",
            maturity=0.45,  # USER PRIOR: user calls TR "unproven"; vault confirms the product exists
            eligible=None,  # "European customers" per vault, but Justin's exact eligibility UNKNOWN
            offers_subscription=True,  # VAULT: in-app IPO subscriptions for European retail
            fee_fixed=1.0,             # VAULT: 1 EUR settlement fee
            fee_ccy="EUR",
            allocation_method="flat pro-rata",  # VAULT + WEB: pro-rata by subscription volume
            # RESEARCH PRIOR (2026-06-09): same deal-level ~5% as Revolut -- no evidence TR fills
            # differently (Euronews: no per-broker rationing data). Equal prior => no fill-based tilt.
            prior_fill_rate=0.05,
            fx_cash_block="EUR settlement (1 EUR fee implies EUR cash); no documented day-1 restriction",
            notes="VAULT + WEB (2026-06-09): in-app IPO subscriptions, official price, pro-rata by "
            "subscription volume, 1 EUR fee; ~30% of SPCX reserved for retail but ~5% float => expect "
            "heavy oversubscription / LOW fill. No max or day-1/flipping restriction documented in TR "
            "launch coverage (user concurs). UNKNOWN: Justin's eligibility, min, exact deadline, "
            "whether full subscription cash is blocked during the book-build.",
        ),
    ]


# --------------------------------------------------------------------------------------
# Recommender math
# --------------------------------------------------------------------------------------
def usable_brokers(brokers: list[Broker]) -> tuple[list[Broker], list[str]]:
    """Drop brokers known-ineligible or known-not-offering; keep UNKNOWNs with a flag."""
    keep: list[Broker] = []
    flags: list[str] = []
    for b in brokers:
        if b.eligible is False:
            flags.append(f"{b.name}: EXCLUDED (known ineligible).")
            continue
        if b.offers_subscription is False:
            flags.append(f"{b.name}: EXCLUDED (known not offering this deal).")
            continue
        keep.append(b)
        if b.eligible is None:
            flags.append(f"{b.name}: eligibility UNKNOWN -- confirm before funding.")
        if b.offers_subscription is None:
            flags.append(f"{b.name}: 'offers subscription for this deal' UNKNOWN -- confirm.")
    return keep, flags


def effective_fill(b: Broker) -> Optional[float]:
    """The fill rate we actually use for this broker: realised if we have it, else the research
    prior, else None (no estimate at all)."""
    return b.realized_fill_rate if b.realized_fill_rate is not None else b.prior_fill_rate


def driver_scores(brokers: list[Broker]) -> tuple[list[float], str]:
    """The per-broker score the split is weighted by -- the heart of the method.

    With flat pro-rata, your shares = fill_rate x subscription, so a broker that fills at a higher
    rate hands you more shares per euro. We weight by the best fill estimate available, in a strict
    preference order:

      1. realised FILL RATE (track record)         -- the real thing, once you've logged deals.
      2. research-based PRIOR fill rate             -- a deal-level estimate from public evidence
                                                       (e.g. SpaceX ~5%); equal across brokers when
                                                       no per-broker evidence exists -> equal split.
      3. maturity PRIOR (gut trust)                 -- last-resort placeholder if there is no fill
                                                       estimate at all.

    We only use a fill-rate tier when EVERY in-scope broker has that estimate, so the comparison
    stays on one scale (mixing a 0.05 fill rate with a 0.90 maturity would be apples-to-oranges).
    """
    effs = [effective_fill(b) for b in brokers]
    if brokers and all(e is not None for e in effs):
        if all(b.realized_fill_rate is not None for b in brokers):
            return effs, "fill-rate (realised track record)"  # type: ignore[return-value]
        return effs, "fill-rate (research-based prior -- no realised data yet)"  # type: ignore[return-value]
    return [b.maturity for b in brokers], "maturity prior (gut; no fill estimate yet)"


def weights_from_scores(scores: list[float], tilt: float) -> list[float]:
    """w_b ∝ score_b ** tilt, renormalised. `tilt` is the CONCENTRATION knob:

      tilt = 0  -> score ignored, EQUAL split (when you have no reason to prefer one, diversify).
      tilt = 1  -> weight PROPORTIONAL to the score (fill rate, or maturity-prior stand-in).
      tilt > 1  -> concentrate harder on the higher-score broker (tilt -> inf puts all at the best).

    So at the default tilt=1 the recommended euro split is literally proportional to estimated
    fill rate: twice the fill rate => twice the money.
    """
    raw = [max(s, 0.0) ** tilt for s in scores]
    tot = sum(raw)
    if tot <= 0:
        n = len(scores) or 1
        return [1.0 / n] * len(scores)
    return [x / tot for x in raw]


def apply_constraints(
    weights: list[float], brokers: list[Broker], budget: float, max_iter: int = 100
) -> tuple[list[float], list[str]]:
    """Water-fill the budget across brokers, clipping to [min, max] and redistributing.

    UNKNOWN min -> treated as 0 (no lower bound). UNKNOWN max -> treated as +inf. So with the
    default all-UNKNOWN config this is a no-op; it becomes meaningful once real limits are set.
    Returns (amounts, warnings).
    """
    warnings: list[str] = []
    mins = [b.min_subscription or 0.0 for b in brokers]
    maxs = [b.max_subscription if b.max_subscription is not None else INF for b in brokers]

    if sum(mins) > budget + 1e-9:
        warnings.append(
            f"INFEASIBLE: sum of broker minimums ({sum(mins):.2f}) exceeds budget ({budget:.2f}). "
            "Drop a broker or raise the budget."
        )
    if sum(maxs) < budget - 1e-9:
        warnings.append(
            f"CAPPED: sum of broker maximums ({sum(maxs):.2f}) is below budget ({budget:.2f}); "
            f"only {sum(maxs):.2f} can be deployed."
        )

    amounts = [w * budget for w in weights]
    for _ in range(max_iter):
        clipped = [min(max(a, lo), hi) for a, lo, hi in zip(amounts, mins, maxs)]
        residual = budget - sum(clipped)
        # brokers free to absorb residual (not pinned at the relevant bound)
        if residual > 1e-9:
            free = [i for i, (a, hi) in enumerate(zip(clipped, maxs)) if a < hi - 1e-9]
        elif residual < -1e-9:
            free = [i for i, (a, lo) in enumerate(zip(clipped, mins)) if a > lo + 1e-9]
        else:
            amounts = clipped
            break
        if not free:
            amounts = clipped
            break
        fw = sum(weights[i] for i in free) or float(len(free))
        for i in free:
            share = (weights[i] / fw) if fw else (1.0 / len(free))
            clipped[i] += residual * share
        amounts = clipped
    return amounts, warnings


@dataclass
class Recommendation:
    capital: float
    tilt: float
    oversubscribe: float
    budget: float                       # = oversubscribe * capital (total subscription)
    weighting_driver: str               # what the split is weighted by (fill-rate or maturity prior)
    rows: list[dict]                    # per-broker: name, weight, amount, fill_rate, maturity...
    expected_fill: Optional[float]      # E[F] = sum(fill_rate * amount) if fill rates known, else None
    worst_case_funded: float            # amount you must fund if phi = 1 everywhere
    warnings: list[str]
    flags: list[str]


def recommend(
    brokers: list[Broker],
    capital: float,
    tilt: float = 1.0,
    oversubscribe: float = 1.0,
) -> Recommendation:
    use, flags = usable_brokers(brokers)
    budget = capital * oversubscribe
    scores, driver = driver_scores(use)
    w = weights_from_scores(scores, tilt)
    amounts, warns = apply_constraints(w, use, budget)
    realised_budget = sum(amounts)
    eff_weights = [a / realised_budget if realised_budget else 0.0 for a in amounts]

    rows: list[dict] = []
    have_means = bool(use) and all(effective_fill(b) is not None for b in use)
    e_fill: Optional[float] = 0.0 if have_means else None
    for b, wt, amt in zip(use, eff_weights, amounts):
        eff = effective_fill(b)
        basis = "realised" if b.realized_fill_rate is not None else ("prior" if b.prior_fill_rate is not None else "—")
        exp = (eff * amt) if eff is not None else None
        if e_fill is not None and exp is not None:
            e_fill += exp
        rows.append(
            {
                "broker": b.name,
                "weight": round(wt, 4),
                "amount": round(amt, 2),
                # the estimated fill rate driving the split (realised mean phi, research prior, or "—")
                "fill_rate": round(eff, 4) if eff is not None else "—",
                "fill_basis": basis,
                "maturity": b.maturity,
                "allocation_method": b.allocation_method or "UNKNOWN",
                "fee": _fee_str(b),
                "min": b.min_subscription if b.min_subscription is not None else "UNKNOWN",
                # None max == no cap enforced by the recommender; shown as "none" not "UNKNOWN"
                "max": b.max_subscription if b.max_subscription is not None else "none",
                "expected_fill_amount": round(exp, 2) if exp is not None else "UNKNOWN",
            }
        )

    if oversubscribe > 1.0 + 1e-9:
        warns.append(
            f"OVER-SUBSCRIPTION: total subscription is {oversubscribe:.2f}x capital "
            f"({budget:.2f} vs {capital:.2f}). If the pro-rata fill lands FULL (phi=1), you must "
            f"fund and hold {realised_budget:.2f} -- far more than your {capital:.2f} budget. "
            "Only over-subscribe if you can fund and want the worst-case full allocation."
        )

    return Recommendation(
        capital=capital,
        tilt=tilt,
        oversubscribe=oversubscribe,
        budget=round(budget, 2),
        weighting_driver=driver,
        rows=rows,
        expected_fill=round(e_fill, 2) if e_fill is not None else None,
        worst_case_funded=round(realised_budget, 2),
        warnings=warns,
        flags=flags,
    )


def _fee_str(b: Broker) -> str:
    if b.fee_fixed is None and b.fee_pct is None:
        return "UNKNOWN"
    if (b.fee_fixed in (0, 0.0)) and (b.fee_pct in (None, 0, 0.0)):
        return "none"
    parts = []
    if b.fee_fixed is not None:
        parts.append(f"{b.fee_fixed:g} {b.fee_ccy or ''}".strip() + " fixed")
    if b.fee_pct is not None:
        parts.append(f"{b.fee_pct:g}% notional")
    return " + ".join(parts)


# --------------------------------------------------------------------------------------
# Rolling fill-rate track record (reads the IPO Subscriptions logs)
# --------------------------------------------------------------------------------------
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _to_float(cell: str) -> Optional[float]:
    cell = cell.strip().replace(",", "")
    if _NUM.match(cell):
        return float(cell)
    return None


def parse_fill_table(text: str) -> list[dict]:
    """Extract realised-fill rows from a markdown pipe table.

    The table is found by a header row containing both 'broker' and 'fill_fraction'. Rows whose
    fill_fraction is non-numeric (TBD / blank / em-dash) are skipped -- those are not yet
    resolved. fill_fraction is computed from filled/requested shares if the column is blank but
    both share columns are numeric.
    """
    lines = text.splitlines()
    header_idx = None
    cols: list[str] = []
    for i, ln in enumerate(lines):
        if "|" in ln and "broker" in ln.lower() and "fill_fraction" in ln.lower():
            cols = [c.strip().lower() for c in ln.strip().strip("|").split("|")]
            header_idx = i
            break
    if header_idx is None:
        return []
    out: list[dict] = []
    for ln in lines[header_idx + 1 :]:
        if "|" not in ln:
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(cols):
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        row = dict(zip(cols, cells))
        broker = row.get("broker", "").strip()
        if not broker or broker.lower() == "broker":
            continue
        ff = _to_float(row.get("fill_fraction", ""))
        if ff is None:
            req = _to_float(row.get("requested_shares", ""))
            fil = _to_float(row.get("filled_shares", ""))
            if req and fil is not None and req > 0:
                ff = fil / req
        if ff is None:
            continue  # unresolved
        out.append({"broker": broker, "fill_fraction": ff})
    return out


def track_record(log_dir: Path) -> dict[str, dict]:
    """Scan IPO Subscriptions/*.md (excluding _template.md, README.md) and aggregate per broker."""
    by_broker: dict[str, list[float]] = {}
    if not log_dir.exists():
        return {}
    for p in sorted(log_dir.glob("*.md")):
        if p.name.lower() in {"_template.md", "readme.md"}:
            continue
        for rec in parse_fill_table(p.read_text(encoding="utf-8")):
            by_broker.setdefault(rec["broker"], []).append(rec["fill_fraction"])
    stats: dict[str, dict] = {}
    for broker, xs in by_broker.items():
        n = len(xs)
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else None
        stats[broker] = {
            "n_deals": n,
            "mean_fill_rate": round(mean, 4),
            "sd_fill_rate": round(sd, 4) if sd is not None else None,
            "min_fill_rate": round(min(xs), 4),
            "max_fill_rate": round(max(xs), 4),
        }
    return stats


def apply_track_record(brokers: list[Broker], stats: dict[str, dict]) -> None:
    """Inject realised mean/sd into the broker config so the recommender can use them."""
    for b in brokers:
        s = stats.get(b.name)
        if s:
            b.realized_fill_rate = s["mean_fill_rate"]
            b.realized_fill_sd = s["sd_fill_rate"]


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def render(rec: Recommendation) -> str:
    L = []
    L.append("=" * 78)
    L.append("EU RETAIL IPO SUBSCRIPTION -- CAPITAL-SPLIT RECOMMENDATION")
    L.append("=" * 78)
    L.append(
        f"capital C = {rec.capital:.2f}   tilt = {rec.tilt:g}   oversubscribe = {rec.oversubscribe:g}"
        f"   total subscription budget = {rec.budget:.2f}"
    )
    L.append(f"WEIGHTED BY: {rec.weighting_driver}")
    L.append("  (split is proportional to this score ^ tilt; at tilt=1, double the score = double the money)")
    L.append("")
    hdr = f"{'broker':<22}{'fill rate':>10}{'basis':>9}{'weight':>9}{'amount':>12}{'fee':>13}"
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in rec.rows:
        fr = r["fill_rate"]
        fr_s = f"{fr*100:>9.1f}%" if isinstance(fr, (int, float)) else f"{fr:>10}"
        L.append(
            f"{r['broker']:<22}{fr_s}{r['fill_basis']:>9}{r['weight']*100:>8.1f}%{r['amount']:>12.2f}{r['fee']:>13}"
        )
    L.append("-" * len(hdr))
    L.append(f"{'TOTAL':<22}{'100.0%':>8}{rec.worst_case_funded:>12.2f}")
    L.append("")
    L.append(
        f"WORST-CASE FUNDED (phi=1 everywhere): {rec.worst_case_funded:.2f}  "
        + ("== your budget (safe)" if abs(rec.worst_case_funded - rec.capital) < 1e-6 else "> your budget")
    )
    if rec.expected_fill is not None:
        basis = "realised track record" if "realised track record" in rec.weighting_driver else "research prior -- rough, wide uncertainty"
        L.append(f"EXPECTED FILL E[F] = sum(fill_rate x amount): {rec.expected_fill:.2f}   [{basis}]")
    else:
        L.append("EXPECTED FILL E[F]: UNKNOWN -- no fill estimate for at least one broker.")
    if rec.flags:
        L.append("")
        L.append("FLAGS (confirm before funding):")
        for f in rec.flags:
            L.append(f"  - {f}")
    if rec.warnings:
        L.append("")
        L.append("WARNINGS:")
        for w in rec.warnings:
            L.append(f"  ! {w}")
    L.append("")
    L.append("HOW TO READ THIS: with flat pro-rata, your shares = fill_rate x subscription, so a")
    L.append("higher fill rate = more shares per euro -> put more money there. The split above is")
    L.append("proportional to the 'fill rate' column (or the maturity prior as a stand-in until you")
    L.append("have real fill data). It is NOT 100% at the best broker because the estimate is noisy")
    L.append("and each broker is a single sealed draw -- spreading hedges that. Log realised fills in")
    L.append("the IPO Subscriptions notes, then re-run with --use-track-record to steer on real data.")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# Self-test (cheap, dependency-free; pytest mirror lives in tests/)
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    # 1) tilt=0 -> equal split (score ignored: diversify when you have no reason to prefer one)
    bs = default_brokers()
    r0 = recommend(bs, 10000, tilt=0.0)
    ws = [row["weight"] for row in r0.rows]
    assert all(abs(w - ws[0]) < 1e-9 for w in ws), f"tilt=0 should be equal weight: {ws}"

    # 2) DEFAULT today: equal research-based fill prior (~5%) on both -> driven by fill prior,
    #    and because the prior is EQUAL the split is 50/50 (no fill-based reason to steer)
    r1 = recommend(default_brokers(), 10000, tilt=2.0)
    assert "research-based prior" in r1.weighting_driver, r1.weighting_driver
    rev = next(x for x in r1.rows if x["broker"] == "Revolut")
    tr = next(x for x in r1.rows if x["broker"] == "Trade Republic")
    assert abs(rev["weight"] - tr["weight"]) < 1e-9, "equal priors must give a 50/50 split"
    assert r1.expected_fill is not None and abs(r1.expected_fill - 500.0) < 1e-6  # 0.05 * 10000

    # 2b) with NO fill estimate at all (clear priors), fall back to maturity -> Revolut > TR
    nofill = default_brokers()
    for b in nofill:
        b.prior_fill_rate = None
    r1b = recommend(nofill, 10000, tilt=2.0)
    assert "maturity prior" in r1b.weighting_driver
    assert next(x for x in r1b.rows if x["broker"] == "Revolut")["weight"] > \
        next(x for x in r1b.rows if x["broker"] == "Trade Republic")["weight"]

    # 3) weights sum to 1, amounts sum to budget (safe default: == capital)
    assert abs(sum(x["weight"] for x in r1.rows) - 1.0) < 1e-9
    assert abs(r1.worst_case_funded - 10000) < 1e-6

    # 4) oversubscription raises worst-case funded above capital and warns
    r2 = recommend(default_brokers(), 10000, tilt=1.0, oversubscribe=1.5)
    assert abs(r2.worst_case_funded - 15000) < 1e-6
    assert any("OVER-SUBSCRIPTION" in w for w in r2.warnings)

    # 5) min/max constraint: a max cap reallocates the excess to the others
    capped = default_brokers()
    capped[0].max_subscription = 1000.0  # Revolut capped at 1000
    r3 = recommend(capped, 10000, tilt=2.0)
    rev3 = next(x for x in r3.rows if x["broker"] == "Revolut")
    assert rev3["amount"] <= 1000.0 + 1e-6, f"cap not respected: {rev3['amount']}"
    assert abs(sum(x["amount"] for x in r3.rows) - 10000) < 1e-6, "budget must still be fully placed"

    # 5b) Revolut's $500 min binds when the budget is small
    r3b = recommend(default_brokers(), 600, tilt=2.0)
    rev3b = next(x for x in r3b.rows if x["broker"] == "Revolut")
    assert rev3b["amount"] >= 500.0 - 1e-6, f"Revolut min not respected: {rev3b['amount']}"

    # 6) THE HEADLINE: once fill rates exist, the split steers toward the higher-fill broker,
    #    proportional to fill rate at tilt=1 (this is the whole point of the tool)
    fr = default_brokers()
    by = {b.name: b for b in fr}
    by["Revolut"].realized_fill_rate = 0.30          # Revolut fills 30%
    by["Trade Republic"].realized_fill_rate = 0.10   # TR fills 10% -> 3x worse
    rfr = recommend(fr, 10000, tilt=1.0)
    assert "fill-rate" in rfr.weighting_driver, rfr.weighting_driver
    rev6 = next(x for x in rfr.rows if x["broker"] == "Revolut")
    tr6 = next(x for x in rfr.rows if x["broker"] == "Trade Republic")
    # weight proportional to fill rate: 0.30/(0.30+0.10)=0.75 vs 0.25
    assert abs(rev6["weight"] - 0.75) < 1e-9 and abs(tr6["weight"] - 0.25) < 1e-9, (rev6, tr6)
    assert rfr.expected_fill is not None and rfr.expected_fill > 0

    # 7) fill-table parser: numeric rows kept, TBD rows skipped, computed when blank
    sample = (
        "| broker | requested_shares | filled_shares | fill_fraction | effective_price | notes |\n"
        "|---|---|---|---|---|---|\n"
        "| Revolut | 100 | 25 | 0.25 | 135 | ok |\n"
        "| Trade Republic | 100 | 10 |  | 135 | computed |\n"
        "| DEGIRO | 100 |  | TBD | 135 | unresolved |\n"
    )
    recs = parse_fill_table(sample)
    got = {r["broker"]: round(r["fill_fraction"], 3) for r in recs}
    assert got == {"Revolut": 0.25, "Trade Republic": 0.1}, got

    print("eu_ipo_capital_split selftest: ALL PASS (9 checks)")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _repo_root() -> Path:
    # scripts/ -> research/ -> polymarket/ -> repo root
    return Path(__file__).resolve().parents[3]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=10000.0, help="total capital C to block (currency).")
    ap.add_argument("--tilt", type=float, default=1.0, help="execution-risk tilt >=0 (0=equal weight).")
    ap.add_argument(
        "--oversubscribe", type=float, default=1.0,
        help="total subscription as a multiple of C (1.0=safe; >1 warns about full-fill risk).",
    )
    ap.add_argument(
        "--prior-fill-rate", type=float, default=None,
        help="override the deal-level research fill-rate prior (same value for all brokers; for a new deal).",
    )
    ap.add_argument(
        "--log-dir", type=str, default=None,
        help="path to the IPO Subscriptions folder (default: <repo>/polymarket/research/notes/overview/market_maps/IPO Subscriptions).",
    )
    ap.add_argument("--track-record", action="store_true", help="print rolling per-broker fill rates and exit.")
    ap.add_argument(
        "--use-track-record", action="store_true",
        help="weight the split by realised fill rates instead of the maturity prior (+ reports E[F]).",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON.")
    ap.add_argument("--selftest", action="store_true", help="run the built-in checks and exit.")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else _repo_root() / "polymarket/research/notes/overview/market_maps/IPO Subscriptions"

    if args.track_record:
        stats = track_record(log_dir)
        if args.json:
            print(json.dumps({"log_dir": str(log_dir), "track_record": stats}, indent=2))
        elif not stats:
            print(f"No resolved fill rows found under: {log_dir}")
            print("(Fill in the 'Realised Fill (per broker)' table in each deal note after allocation.)")
        else:
            print(f"Rolling per-broker fill-rate track record  (source: {log_dir})")
            print(f"{'broker':<22}{'n':>4}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
            for broker, s in sorted(stats.items()):
                sd = f"{s['sd_fill_rate']:.3f}" if s["sd_fill_rate"] is not None else "  -- "
                print(
                    f"{broker:<22}{s['n_deals']:>4}{s['mean_fill_rate']:>8.3f}{sd:>8}"
                    f"{s['min_fill_rate']:>8.3f}{s['max_fill_rate']:>8.3f}"
                )
        return 0

    brokers = default_brokers()
    if args.prior_fill_rate is not None:
        for b in brokers:
            b.prior_fill_rate = args.prior_fill_rate
    if args.use_track_record:
        apply_track_record(brokers, track_record(log_dir))

    rec = recommend(brokers, args.capital, tilt=args.tilt, oversubscribe=args.oversubscribe)
    if args.json:
        print(json.dumps(asdict(rec), indent=2, default=str))
    else:
        print(render(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
