"""Data-channel snapshot — the OFFLINE INGEST BOUNDARY for the live Fed market.

Computes `p_struct` for the markets in `config.DATA_CHANNEL_MARKETS` under the
reaction function locked in [[newsagent_data_channel_scoping]] § 4f (DC-1…DC-8 +
Amendment 1 + Amendment 2), and writes it to JSON. Nothing else in the pipeline
computes it, and `newsagent/*` only ever READS the JSON this script writes.

**The AGPL boundary (Justin acknowledged 2026-08-24, sign-off row 2).** OpenBB is
AGPL-3.0-only. It is installed in the research venv as row 1 authorises "when the
build starts", and it may be imported HERE — an offline snapshot script whose
output is data — but **never** from `newsagent/*`, which is the code that renders
a public page. `tests/test_newsagent_datachannel.py::test_newsagent_never_imports_openbb`
enforces that boundary rather than trusting it.

**Measured 2026-08-24, and it matters: OpenBB cannot serve this channel's inputs.**
`obb.economy.fred_series` exposes `symbol / start_date / end_date / limit /
provider` and **no realtime (vintage) parameters at all**. DC-7 requires ALFRED
realtime vintages — "current FRED values for revised series must NOT be used to
reconstruct an as-of state; ALFRED vintages or nothing" — so the vintage-critical
series are pulled here by direct ALFRED call, exactly as the dry run and v2/v3
did and proved lookahead-free. OpenBB is therefore installed but **not on the
critical path**; see the findings note for the recommendation.

Run: PYTHONPATH=. uv run python scripts/newsagent_datachannel_snapshot.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newsagent_datachannel_dryrun import (  # noqa: E402
    CACHE, MC_DRAWS, P_CLIP, SEED, SIGMA_MOM, KAPPA_PI, KAPPA_U,
    as_of, fred_today, load_nowcasts, month_add, nowcast_asof,
)
from newsagent_datachannel_v3 import (  # noqa: E402
    B1A, FOMC, W_NON, W_SEP, vintage_series,
)
from newsagent_datachannel_v2 import base_rate_all_years  # noqa: E402
from newsagent import config  # noqa: E402

OUT_DIR = config.ROOT / "data" / "newsagent" / "datachannel"

# slug -> which meeting this market resolves on. Declared, not inferred: a market
# is only in this map when a human has checked that its resolution criterion is
# "no change at THIS meeting".
MARKET_MEETINGS = {
    "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615":
        {"decision": date(2026, 9, 16), "sep": True, "sep_year": 2026},
}


def p_struct(meeting: dict, t: date, nc: dict, logit_h_base: float) -> dict | None:
    """The locked reaction function at date t, from vintage-only inputs.

    Amendment 1 (DC-3a allocate / DC-3b all-years base rate) as amended by
    Amendment 2 (W_SEP = 1.0). Every constant is imported from the v3 module so
    there is exactly one definition of each and a drift between the retro-test
    and the live channel is impossible.
    """
    sep_year = meeting["sep_year"]
    key = f"{sep_year}-01-01"
    start = t - timedelta(days=45)
    ser = {sid: vintage_series(sid, f"{start.year - 3}-01-01", start.isoformat(),
                               t.isoformat())
           for sid in ("DFEDTARU", "DFEDTARL", "PCEPILFE", "UNRATE",
                       "FEDTARMD", "JCXFEMD", "UNRATEMD")}
    u_pub, pce_pub = as_of(ser["UNRATE"], t), as_of(ser["PCEPILFE"], t)
    up, lo = as_of(ser["DFEDTARU"], t), as_of(ser["DFEDTARL"], t)
    sep_r = as_of(ser["FEDTARMD"], t).get(key)
    sep_pi = as_of(ser["JCXFEMD"], t).get(key)
    sep_u = as_of(ser["UNRATEMD"], t).get(key)
    if not (u_pub and pce_pub and up and lo and sep_r and sep_pi and sep_u):
        return None
    dk = [d for d in up if d <= t.isoformat()]
    lk = [d for d in lo if d <= t.isoformat()]
    if not dk or not lk:
        return None
    mid = (up[max(dk)] + lo[max(lk)]) / 2.0

    last_m = max(pce_pub)
    ly, lm = int(last_m[:4]), int(last_m[5:7])
    k = (t.year * 12 + t.month) - (ly * 12 + lm)
    idx = pce_pub[last_m]
    chain = []
    for j in range(1, k + 1):
        ym = month_add((ly, lm), j)
        v = nowcast_asof(nc, ym, "Core PCE Inflation", t)
        src = "nowcast"
        if v is None:
            pk, bk = month_add((ly, lm), j - 1), month_add((ly, lm), j - 2)
            pkey, bkey = f"{pk[0]:04d}-{pk[1]:02d}-01", f"{bk[0]:04d}-{bk[1]:02d}-01"
            if pkey in pce_pub and bkey in pce_pub:
                v, src = (pce_pub[pkey] / pce_pub[bkey] - 1) * 100, "prev_print"
            else:
                v, src = 0.0, "flat"
        idx *= (1 + v / 100.0)
        chain.append({"ym": f"{ym[0]}-{ym[1]:02d}", "mom_pct": round(v, 4), "src": src})
    cur = month_add((ly, lm), k)
    bkey = f"{cur[0] - 1:04d}-{cur[1]:02d}-01"
    if bkey not in pce_pub:
        return None
    pi_center = (idx / pce_pub[bkey] - 1) * 100.0
    u_last = u_pub[max(u_pub)]

    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(MC_DRAWS // 2)
    z = np.concatenate([z, -z])
    sigma = SIGMA_MOM * math.sqrt(max(k, 1))
    pi_draws = pi_center + sigma * z
    r_des = sep_r + KAPPA_PI * (pi_draws - sep_pi) - KAPPA_U * (u_last - sep_u)
    G = np.abs(r_des - mid) / 0.25

    rem = [(d, s) for d, s in FOMC.get(sep_year, []) if d >= t]
    ws = [W_SEP if s else W_NON for _, s in rem] or [1.0]
    share1 = ws[0] / sum(ws)
    zz = logit_h_base + B1A * (G * share1)
    p = float(np.mean(np.clip(1.0 - 1.0 / (1.0 + np.exp(-zz)), *P_CLIP)))

    r_des_pt = sep_r + KAPPA_PI * (pi_center - sep_pi) - KAPPA_U * (u_last - sep_u)
    return {
        "p_struct_pct": round(p * 100, 1),
        "asof": t.isoformat(),
        "meeting": meeting["decision"].isoformat(),
        "meetings_remaining": len(rem),
        "share_this_meeting": round(share1, 4),
        "gap_pp": round(r_des_pt - mid, 4),
        "G_clicks": round(abs(r_des_pt - mid) / 0.25, 4),
        "mu1_clicks": round(abs(r_des_pt - mid) / 0.25 * share1, 4),
        "inputs": {"core_pce_yoy_pct": round(pi_center, 3), "unrate_pct": u_last,
                   "sep_rate_median_pct": sep_r, "sep_core_pce_pct": sep_pi,
                   "sep_unrate_pct": sep_u, "target_midpoint_pct": mid,
                   "unpublished_months_chained": k, "last_published_pce": last_m,
                   "nowcast_chain": chain},
        "method": "news+data",
        "rule": "DC-3a allocate (W_SEP=1.0) + DC-3b all-years base rate, "
                "§ 4f Amendment 1 + Amendment 2, locked 2026-08-24",
    }


def market_implied(meeting: date) -> dict:
    """Atlanta Fed MPT — DISPLAY ONLY (DC-6). Never an input, anywhere.

    **The MPT stops quoting a window once that window opens** (dry-run § 7): the
    window containing a meeting goes dark from roughly the moment it starts, so a
    panel that assumes a continuous series will have a hole exactly when the
    meeting gets interesting. This returns a structured ABSENCE rather than
    raising, and the card renders the absence in words.
    """
    return {"available": False,
            "why": ("the Atlanta Fed Market Probability Tracker stops quoting a "
                    "3-month window once that window opens, so no market-implied "
                    "context is available for this meeting"),
            "note": "display-only under DC-6; never an input to p_struct."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    # Effective as-of: never later than the newest vintage FRED will serve. Moving
    # BACKWARD is always safe for lookahead discipline — an earlier as-of can only
    # ever see less. Moving forward is what breaks the run, so we never do it.
    asof = fred_today(args.date)
    if asof != args.date:
        print(f"  as-of clamped {args.date} -> {asof} (FRED's own clock; the "
              f"snapshot is stamped with the date it can actually justify)")
    t = date.fromisoformat(asof)

    hold_all, meta = base_rate_all_years()
    h_base = 1.0 - hold_all
    logit_h_base = math.log(h_base / (1 - h_base))
    nc = load_nowcasts()

    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "asof": asof, "asof_requested": args.date,
           "base_rate": {"hold_all_years": round(hold_all, 4), **meta},
           "constants": {"W_SEP": W_SEP, "W_NON": W_NON, "B1A": B1A,
                         "KAPPA_PI": KAPPA_PI, "KAPPA_U": KAPPA_U,
                         "SIGMA_MOM": SIGMA_MOM, "MC_DRAWS": MC_DRAWS, "SEED": SEED},
           "markets": {}}
    for slug, mk in MARKET_MEETINGS.items():
        rec = p_struct(mk, t, nc, logit_h_base)
        if rec is None:
            print(f"  {slug[:56]}: inputs incomplete as of {asof} — skipped")
            continue
        rec["market_implied"] = market_implied(mk["decision"])
        out["markets"][slug] = rec
        print(f"  {slug[:56]}\n    p_struct={rec['p_struct_pct']}%  "
              f"gap={rec['gap_pp']:+.3f}pp  G={rec['G_clicks']:.2f}cl  "
              f"share={rec['share_this_meeting']:.3f}  mu1={rec['mu1_clicks']:.3f}  "
              f"({rec['meetings_remaining']} meetings left in {mk['sep_year']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"p_struct_{asof}.json").write_text(json.dumps(out, indent=1))
    latest = OUT_DIR / "p_struct_latest.json"
    # A run that produced NO market must not overwrite a good snapshot with an empty
    # one. The read side already degrades gracefully on a stale snapshot (7-day
    # bound, reason printed); it has no defence against a fresh-but-empty one, which
    # silently strips the live card's anchor instead of ageing it.
    if not out["markets"] and latest.exists():
        print(f"\n  REFUSING to overwrite {latest.name}: this run produced no market "
              f"(the previous snapshot stands and ages under the declared 7-day bound)")
        return 1
    latest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
