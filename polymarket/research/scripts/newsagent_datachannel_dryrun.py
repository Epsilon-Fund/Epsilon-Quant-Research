"""Data-channel DRY RUN — score the resolved July-2026 Fed market with vintage-only inputs.

Retrospectively computes p_struct(t) for
`will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting`
over 2026-06-03 … 2026-07-29 under the DECLARED reaction function of
newsagent_data_channel_scoping.md § 4f (DC-1 … DC-8, LOCKED 2026-08-24) and the
pre-registration locked to the session scratchpad BEFORE any number existed
(reproduced verbatim in the findings note).

Nothing here is fitted on this market. Every constant comes from the pre-registration.

Lookahead discipline (DC-7): every input is point-in-time.
  - macro levels/rates/SEP medians via the FRED **ALFRED** realtime interface
    (one call per series over the window; for each t we take the row whose
    realtime interval contains t) — so a revision or an SEP published later is
    invisible at t, verified live (FEDTARMD-2026 reads 3.4 on 2026-06-10 and 3.8
    on 2026-06-18, i.e. the June SEP appears only after it was published);
  - the Cleveland Fed inflation nowcast carries DAILY vintages, so the current
    month's inflation estimate at t is the vintage dated <= t;
  - the Atlanta Fed MPT is fetched for DISPLAY ONLY (DC-6) and never touches
    p_struct.

Run (from polymarket/research/):
  PYTHONPATH=. uv run python scripts/newsagent_datachannel_dryrun.py [--chart]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newsagent import config  # noqa: E402  (loads the git-ignored .env)
from newsagent.config import CSV_OUT, DATA, ROOT  # noqa: E402

CACHE = ROOT / "data" / "newsagent" / "datachannel"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"
SLUG = "will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting"

# ---- pre-registered constants (§ 4 of the lock; NOT fitted) ---------------------
WIN_START, WIN_END = date(2026, 6, 3), date(2026, 7, 29)
KAPPA_PI, KAPPA_U = 0.5, 0.5        # borrowed Taylor coefficients
B1 = -3.0                            # declared slope on |per-meeting clicks|
SIGMA_MOM = 0.063                    # core-PCE MoM surprise RMSE (§ 3 M4, primary window)
P_CLIP = (0.02, 0.98)
MC_DRAWS = 20_000
SEED = 0
ZLB_YEARS = set(range(2009, 2016)) | {2020, 2021}
BASE_YEARS = [y for y in range(1994, 2026) if y not in ZLB_YEARS]
MEETINGS_PER_YEAR = 8
# 2026 FOMC calendar, parsed from federalreserve.gov (published years in advance)
FOMC_2026 = [date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
             date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]
MEETING = date(2026, 7, 29)
OUTCOME = 1                          # resolved YES (no change), verified on Gamma

FRED_API = "https://api.stlouisfed.org/fred/"
NOWCAST_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
               "inflationnowcasting/nowcast_month.json")
MPT_URL = ("https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
           "cenfis/market-probability-tracker/mpt_histdata.xlsx")


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "epsilon-research/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


# FRED's server clock runs on US Central time, so between UTC midnight and Central
# midnight a UTC "today" is one day AHEAD of the date FRED will accept, and every
# realtime_end=today request 400s with a message naming the server's own date. Left
# unhandled this makes the live data channel unrefreshable for a ~5-6 hour window
# every single day — the snapshot dies, `p_struct` goes stale, and after 7 days the
# card silently degrades to its onboarding prior. Parsed rather than hardcoded: the
# server tells us its date, so no timezone assumption is baked in here.
_LAST_SERVER_TODAY: str | None = None   # set by fred() when the clamp below fires
_FRED_TODAY_MEMO: dict[str, str] = {}   # one probe per requested date per process
_RT_END_TOO_LATE = re.compile(
    r"realtime_(?:start|end) can not be after today's date "
    r"\((\d{4}-\d{2}-\d{2})\)")


def fred(path: str, **params) -> dict:
    key = os.environ.get("OPENBB_FRED_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENBB_FRED_API_KEY missing — put it in polymarket/research/.env")
    p = dict(params)
    p.update({"api_key": key, "file_type": "json"})
    try:
        return json.loads(_get(FRED_API + path + "?" + urllib.parse.urlencode(p)))
    except urllib.error.HTTPError as e:
        if e.code != 400 or "realtime_end" not in p:
            raise
        m = _RT_END_TOO_LATE.search(e.read().decode("utf-8", "replace"))
        if not m:
            raise
        server_today = m.group(1)
        global _LAST_SERVER_TODAY
        _LAST_SERVER_TODAY = server_today
        if server_today >= str(p["realtime_end"]):
            raise                      # not the clock-skew case — do not mask it
        p["realtime_end"] = server_today
        print(f"  FRED clock skew: realtime_end clamped to the server's own date "
              f"{server_today} (vintage discipline unaffected — clamping the end of a "
              f"realtime window can only ever REMOVE a later vintage, never add one)")
        return json.loads(_get(FRED_API + path + "?" + urllib.parse.urlencode(p)))


def fred_today(requested: str) -> str:
    """The latest date ALFRED will actually serve, at or before `requested`.

    Needed because FRED runs on US Central: for a ~5-6 hour window after UTC
    midnight, a UTC "today" is a date FRED does not have, and asking for it yields
    rows whose realtime spans all END the previous day — so as_of(t) matches
    nothing and EVERY vintage input comes back empty. Clamping the HTTP call alone
    is not enough; the as-of date the whole snapshot is computed at has to move
    too, or the channel is unusable for part of every day.

    One cheap probe. Returns `requested` unchanged whenever FRED accepts it, so on
    a normal run this costs one small request and changes nothing. Note we read the
    clamp FLAG rather than catching an exception: fred() self-heals the 400, so the
    probe call succeeds either way and the only evidence of skew is the flag."""
    if requested in _FRED_TODAY_MEMO:
        return _FRED_TODAY_MEMO[requested]
    global _LAST_SERVER_TODAY
    _LAST_SERVER_TODAY = None
    try:
        # realtime_start deliberately well in the past: a future start trips a
        # DIFFERENT 400 and would tell us nothing about the end bound we care about.
        fred("series/observations", series_id="DFEDTARU", limit=1,
             observation_start="2020-01-01", realtime_start="2020-01-01",
             realtime_end=requested)
    except Exception:
        pass
    _FRED_TODAY_MEMO[requested] = _LAST_SERVER_TODAY or requested
    return _FRED_TODAY_MEMO[requested]


def vintage_series(series_id: str, obs_start: str, rt_start: str, rt_end: str) -> list[dict]:
    """ALFRED rows with their realtime intervals (cached).

    rt_end is resolved to the newest date FRED will actually serve BEFORE the cache
    filename is built. Doing it here rather than inside fred() matters: if the HTTP
    layer silently clamps while the filename still claims the later date, the cache
    is POISONED — every subsequent run reads rows whose realtime spans all end a day
    early, as_of(t) matches nothing, and the channel reports "inputs incomplete"
    forever with no error to look at. Resolve once, key everything on the result."""
    rt_end = fred_today(rt_end)
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"alfred_{series_id}_{rt_start}_{rt_end}.json"
    if f.exists():
        return json.loads(f.read_text())
    rows = fred("series/observations", series_id=series_id, observation_start=obs_start,
                realtime_start=rt_start, realtime_end=rt_end)["observations"]
    f.write_text(json.dumps(rows))
    return rows


def as_of(rows: list[dict], t: date) -> dict[str, float]:
    """{observation date -> value} as PUBLISHED on t (the row whose realtime span covers t)."""
    out: dict[str, float] = {}
    for o in rows:
        if o["value"] in (".", "", None):
            continue
        if o["realtime_start"] <= t.isoformat() <= o["realtime_end"]:
            out[o["date"]] = float(o["value"])
    return out


def load_nowcasts() -> dict[tuple[int, int], dict[str, list[tuple[date, float]]]]:
    """Cleveland Fed JSON -> {(year, month): {series: [(vintage_date, value_mom_pct)]}}."""
    f = CACHE / "cleveland_nowcast_month.json"
    blob = json.loads(f.read_text()) if f.exists() else json.loads(_get(NOWCAST_URL, 180))
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(blob))
    out: dict[tuple[int, int], dict[str, list]] = {}
    for entry in blob:
        m = re.match(r"^(\d{4})-(\d{1,2})$", entry["chart"].get("subcaption", "").strip())
        if not m:
            continue
        ty, tm = int(m.group(1)), int(m.group(2))
        labels = [c.get("label", "") for c in entry["categories"][0]["category"]]
        per: dict[str, list] = {}
        for ds in entry.get("dataset", []):
            name = ds.get("seriesname", "")
            pts = []
            for i, pt in enumerate(ds.get("data", [])):
                v = pt.get("value")
                if v in (None, "") or i >= len(labels):
                    continue
                lm = re.match(r"^(\d{2})/(\d{2})$", labels[i])
                if not lm:
                    continue
                mm, dd = int(lm.group(1)), int(lm.group(2))
                yy = ty if mm == tm else (ty + 1 if mm < tm else ty)
                try:
                    pts.append((date(yy, mm, dd), float(v)))
                except ValueError:
                    continue
            per[name] = sorted(pts)
        out[(ty, tm)] = per
    return out


def nowcast_asof(nc: dict, ym: tuple[int, int], series: str, t: date) -> float | None:
    pts = nc.get(ym, {}).get(series, [])
    ok = [v for d, v in pts if d <= t]
    return ok[-1] if ok else None


def month_add(ym: tuple[int, int], k: int) -> tuple[int, int]:
    m0 = ym[0] * 12 + (ym[1] - 1) + k
    return m0 // 12, m0 % 12 + 1


def base_rate() -> tuple[float, dict]:
    """Share of scheduled FOMC meetings with NO target change, declared reference sample."""
    rows_u = vintage_series("DFEDTARU", "2008-12-01", "2026-08-24", "2026-08-24")
    rows_old = vintage_series("DFEDTAR", "1994-01-01", "2026-08-24", "2026-08-24")
    changes: set[date] = set()
    for rows in (rows_old, rows_u):
        prev = None
        for o in rows:
            if o["value"] in (".", "", None):
                continue
            v, d = float(o["value"]), datetime.fromisoformat(o["date"]).date()
            if prev is not None and abs(v - prev) > 1e-9 and d.year in BASE_YEARS:
                changes.add(d)
            prev = v
    n_meetings = MEETINGS_PER_YEAR * len(BASE_YEARS)
    base = 1.0 - len(changes) / n_meetings
    return base, {"n_change_dates": len(changes), "n_meetings": n_meetings,
                  "years": len(BASE_YEARS), "excluded": sorted(ZLB_YEARS)}


def p_hold(m_clicks: np.ndarray | float, b0: float) -> np.ndarray | float:
    z = b0 + B1 * np.abs(m_clicks)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, *P_CLIP)


def build() -> dict:
    nc = load_nowcasts()
    rt0, rt1 = WIN_START.isoformat(), WIN_END.isoformat()
    ser = {
        "DFEDTARU": vintage_series("DFEDTARU", "2026-01-01", rt0, rt1),
        "DFEDTARL": vintage_series("DFEDTARL", "2026-01-01", rt0, rt1),
        "PCEPILFE": vintage_series("PCEPILFE", "2024-06-01", rt0, rt1),
        "UNRATE": vintage_series("UNRATE", "2026-01-01", rt0, rt1),
        "FEDTARMD": vintage_series("FEDTARMD", "2026-01-01", rt0, rt1),
        "JCXFEMD": vintage_series("JCXFEMD", "2026-01-01", rt0, rt1),
        "UNRATEMD": vintage_series("UNRATEMD", "2026-01-01", rt0, rt1),
    }
    b0_base, base_meta = base_rate()
    b0 = math.log(b0_base / (1 - b0_base))
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(MC_DRAWS // 2)
    z = np.concatenate([z, -z])                      # antithetic

    rows = []
    t = WIN_START
    while t <= WIN_END:
        u_pub = as_of(ser["UNRATE"], t)
        pce_pub = as_of(ser["PCEPILFE"], t)
        up = as_of(ser["DFEDTARU"], t)
        lo = as_of(ser["DFEDTARL"], t)
        sep_r = as_of(ser["FEDTARMD"], t).get("2026-01-01")
        sep_pi = as_of(ser["JCXFEMD"], t).get("2026-01-01")
        sep_u = as_of(ser["UNRATEMD"], t).get("2026-01-01")
        if not (u_pub and pce_pub and up and lo and sep_r and sep_pi and sep_u):
            t += timedelta(days=1)
            continue
        # policy state: last daily observation on/before t
        dkeys = [d for d in up if d <= t.isoformat()]
        mid = (up[max(dkeys)] + lo[max(dkeys)]) / 2.0

        # --- core PCE y/y, vintage + nowcast chain ---------------------------
        last_m = max(pce_pub)                       # last PUBLISHED month
        ly, lm = int(last_m[:4]), int(last_m[5:7])
        cy, cm = t.year, t.month                    # current calendar month
        k = (cy * 12 + cm) - (ly * 12 + lm)         # unpublished months to chain
        idx = pce_pub[last_m]
        chain, mom_used = [], []
        for j in range(1, k + 1):
            ym = month_add((ly, lm), j)
            v = nowcast_asof(nc, ym, "Core PCE Inflation", t)
            src = "nowcast"
            if v is None:                            # DC-2 fallback: previous print
                prev_key = f"{month_add((ly, lm), j - 1)[0]:04d}-{month_add((ly, lm), j - 1)[1]:02d}-01"
                base_key = f"{month_add((ly, lm), j - 2)[0]:04d}-{month_add((ly, lm), j - 2)[1]:02d}-01"
                if prev_key in pce_pub and base_key in pce_pub:
                    v = (pce_pub[prev_key] / pce_pub[base_key] - 1) * 100
                    src = "prev_print"
                else:
                    v, src = 0.0, "flat"
            idx *= (1 + v / 100.0)
            chain.append({"ym": f"{ym[0]}-{ym[1]:02d}", "mom_pct": round(v, 4), "src": src})
            mom_used.append(v)
        cur_ym = month_add((ly, lm), k)
        base_key = f"{cur_ym[0] - 1:04d}-{cur_ym[1]:02d}-01"
        if base_key not in pce_pub:
            t += timedelta(days=1)
            continue
        pi_center = (idx / pce_pub[base_key] - 1) * 100.0

        u_last = u_pub[max(u_pub)]
        n_rem = sum(1 for d in FOMC_2026 if d >= t)

        # --- DC-2: integrate the unpublished-print uncertainty ---------------
        sigma = SIGMA_MOM * math.sqrt(max(k, 1))
        pi_draws = pi_center + sigma * z
        r_des = sep_r + KAPPA_PI * (pi_draws - sep_pi) - KAPPA_U * (u_last - sep_u)
        m_clicks = ((r_des - mid) / n_rem) / 0.25
        p_struct = float(np.mean(p_hold(m_clicks, b0)))

        # point estimate (no integration) — reported as a diagnostic only
        r_des_pt = sep_r + KAPPA_PI * (pi_center - sep_pi) - KAPPA_U * (u_last - sep_u)
        m_pt = ((r_des_pt - mid) / n_rem) / 0.25

        rows.append({
            "date": t.isoformat(), "p_struct": round(p_struct, 4),
            "p_point": round(float(p_hold(m_pt, b0)), 4),
            "mid_target": mid, "pi_core_pce_yoy": round(pi_center, 3),
            "unrate": u_last, "sep_rate": sep_r, "sep_core_pce": sep_pi,
            "sep_unrate": sep_u, "r_desired": round(r_des_pt, 4),
            "gap_pp": round(r_des_pt - mid, 4), "n_rem": n_rem,
            "m_clicks": round(m_pt, 4), "sigma_pi": round(sigma, 4),
            "k_unpublished": k, "last_pce_month": last_m,
            "chain": chain,
        })
        t += timedelta(days=1)
    return {"rows": rows, "b0": b0, "base_rate": b0_base, "base_meta": base_meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    out = build()
    rows, base = out["rows"], out["base_rate"]
    print(f"base rate of 'no change' ({out['base_meta']['years']} non-ZLB years, "
          f"{out['base_meta']['n_change_dates']} change dates / {out['base_meta']['n_meetings']} "
          f"meetings): {base:.4f}  -> b0 = {out['b0']:.4f}")
    print(f"days computed: {len(rows)}  ({rows[0]['date']} … {rows[-1]['date']})")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "dryrun_pstruct.json").write_text(json.dumps(out, indent=1))
    for r in rows[::7] + [rows[-1]]:
        print(f"  {r['date']}  p_struct={r['p_struct']:.3f}  pi={r['pi_core_pce_yoy']:.2f}  "
              f"u={r['unrate']}  SEP={r['sep_rate']}  mid={r['mid_target']}  "
              f"gap={r['gap_pp']:+.3f}  n_rem={r['n_rem']}  m={r['m_clicks']:+.3f}  k={r['k_unpublished']}")

    sc = score(out)
    mpt = mpt_series(WIN_START, WIN_END)                       # window CONTAINING July
    mpt_next = mpt_series(WIN_START, WIN_END, ref="2026-09-16")  # the NEXT window
    out["score"], out["mpt"], out["mpt_next"] = sc, mpt, mpt_next
    (CACHE / "dryrun_pstruct.json").write_text(json.dumps(out, indent=1))

    a, b, c = sc["a"], sc["b"], sc["c"]
    print("\n--- (a) accuracy on the news-FV intersection "
          f"({a['n_intersection']} days, {a['first']} … {a['last']}); outcome YES ---")
    print(f"  p_struct        {a['brier_p_struct']:.4f}")
    print(f"  news FV         {a['brier_news_fv']:.4f}   a1 {'PASS' if a['a1_pass'] else 'FAIL'}")
    print(f"  prior-only      {a['brier_prior_only']:.4f}   a2 {'PASS' if a['a2_pass'] else 'FAIL'}")
    print(f"  base-rate-only  {a['brier_base_rate_only']:.4f}   (a3, no bar)")
    print(f"  PM mid (ctx)    {a['brier_mid_context']:.4f}")
    print(f"  p_struct over its own {a['n_full_window']}-day window: {a['brier_p_struct_full_window']:.4f}")
    print(f"\n--- (b) directional sanity: {b['n_directional']} directional release days, "
          f"{b['violations']} violations -> {'PASS' if b['b_pass'] else 'FAIL'} ---")
    for ch in b["checks"]:
        print(f"  {ch['date']}  {ch['kind']:36s} dp={ch['delta_p']:+.4f}  "
              f"implied={ch['implied_direction']}  {'VIOLATION' if ch['violation'] else 'ok'}")
    print(f"\n--- (c) stability: max daily |dlogit| = {c['max_daily_abs_dlogit']:.3f} "
          f"on {c['max_daily_on']} (cap reference 1.5) -> {'FLAG' if c['c_flag'] else 'no flag'}; "
          f"end-to-end |dlogit| = {c['total_abs_dlogit_end_to_end']:.3f}")
    if mpt:
        print(f"\n--- MPT (DISPLAY ONLY, lower bound): {mpt[0]['date']} p_hold_lb="
              f"{mpt[0]['p_hold_lb']:.3f} … {mpt[-1]['date']} p_hold_lb={mpt[-1]['p_hold_lb']:.3f} "
              f"(n={len(mpt)}, window 2026-06-17 -> 2026-09-16, range {mpt[-1]['target_range']})")
        print(f"    NOTE: the window containing the July meeting stops being quoted on "
              f"{mpt[-1]['date']} — the tracker rolls forward once a window opens.")
    if mpt_next:
        print(f"    next window (2026-09-16 -> 2026-12-16), display only: "
              f"{mpt_next[0]['date']} p_hold_lb={mpt_next[0]['p_hold_lb']:.3f} … "
              f"{mpt_next[-1]['date']} p_hold_lb={mpt_next[-1]['p_hold_lb']:.3f} (n={len(mpt_next)})")
    print(f"\n=== VERDICT: {sc['verdict']} ===")

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    fv, p0 = news_fv()
    with open(CSV_OUT / "newsagent_datachannel_dryrun_trajectory.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["date", "p_struct", "p_struct_point_estimate", "news_fv", "pm_mid",
                    "mpt_p_hold_lower_bound", "base_rate_only", "pi_core_pce_yoy", "unrate",
                    "sep_rate", "mid_target", "gap_pp", "n_rem", "m_clicks", "k_unpublished"])
        mptd = {m["date"]: m for m in mpt}
        for r in rows:
            d = r["date"]
            w.writerow([d, r["p_struct"], r["p_point"],
                        fv[d]["fv_pct"] / 100.0 if d in fv else "",
                        fv[d]["mid_pct"] / 100.0 if d in fv and fv[d].get("mid_pct") is not None else "",
                        mptd[d]["p_hold_lb"] if d in mptd else "",
                        round(out["base_rate"], 4), r["pi_core_pce_yoy"], r["unrate"],
                        r["sep_rate"], r["mid_target"], r["gap_pp"], r["n_rem"],
                        r["m_clicks"], r["k_unpublished"]])
    with open(CSV_OUT / "newsagent_datachannel_dryrun_criteria.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["criterion", "value", "comparator", "pass"])
        w.writerow(["a1 Brier p_struct vs news FV", a["brier_p_struct"], a["brier_news_fv"], a["a1_pass"]])
        w.writerow(["a2 Brier p_struct vs prior-only", a["brier_p_struct"], a["brier_prior_only"], a["a2_pass"]])
        w.writerow(["a3 Brier p_struct vs base-rate-only", a["brier_p_struct"], a["brier_base_rate_only"], "no bar"])
        w.writerow(["b directional violations", b["violations"], 0, b["b_pass"]])
        w.writerow(["c max daily |dlogit|", c["max_daily_abs_dlogit"], 1.5, not c["c_flag"]])
        w.writerow(["verdict", sc["verdict"], "", ""])
    print(f"csv -> {CSV_OUT / 'newsagent_datachannel_dryrun_trajectory.csv'}")

    if args.chart:
        chart(out)
    return 0


def chart(out: dict) -> None:
    import matplotlib          # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt   # noqa: PLC0415
    rows = out["rows"]
    fv, p0 = news_fv()
    mptd = {m["date"]: m["p_hold_lb"] for m in out.get("mpt", [])}
    xs = [datetime.fromisoformat(r["date"]).date() for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5]})
    ax.plot(xs, [r["p_struct"] * 100 for r in rows], color="#cc5c44", lw=2.2,
            label="p_struct — declared structural method (vintage-only)")
    ax.plot(xs, [out["base_rate"] * 100] * len(xs), color="#8a8578", lw=1.2, ls=":",
            label=f"base-rate-only ({out['base_rate']*100:.1f}%)")
    fx = [x for x in xs if x.isoformat() in fv]
    ax.plot(fx, [fv[x.isoformat()]["fv_pct"] for x in fx], color="#3d6b8e", lw=2.0,
            label="news FV (what the Observatory published)")
    mx = [x for x in xs if x.isoformat() in fv and fv[x.isoformat()].get("mid_pct") is not None]
    ax.plot(mx, [fv[x.isoformat()]["mid_pct"] for x in mx], color="#4a4a46", lw=1.6, ls="--",
            label="Polymarket mid (context)")
    px = [x for x in xs if x.isoformat() in mptd]
    ax.plot(px, [mptd[x.isoformat()] * 100 for x in px], color="#6b8e3d", lw=2.0, ls="-.",
            marker="o", ms=3,
            label="MPT, window CONTAINING July (Jun17–Sep16) — lower bound, display only")
    mnd = {m["date"]: m["p_hold_lb"] for m in out.get("mpt_next", [])}
    nx = [x for x in xs if x.isoformat() in mnd]
    ax.plot(nx, [mnd[x.isoformat()] * 100 for x in nx], color="#9bb06b", lw=1.3, ls=":",
            label="MPT, NEXT window (Sep16–Dec16) — lower bound, display only")
    ax.axhline(100, color="#2e7d32", lw=1.0, alpha=0.5)
    ax.text(xs[1], 96.5, "outcome: YES (no change) — 2026-07-29", fontsize=8, color="#2e7d32")
    for d, lab in [("2026-06-17", "June SEP (median 3.4->3.8)"), ("2026-06-25", "May core PCE"),
                   ("2026-07-02", "June jobs"), ("2026-07-29", "FOMC decision")]:
        dd = datetime.fromisoformat(d).date()
        if xs[0] <= dd <= xs[-1]:
            ax.axvline(dd, color="#b8b7ad", lw=0.8, ls=":")
            ax.text(dd, 8, lab, rotation=90, fontsize=7, color="#5a5a55", va="bottom")
    ax.set_ylabel("P(no change at the July 2026 meeting), %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_title("Data-channel dry run — July-2026 Fed market, vintage-only inputs (n=1 falsifier)")
    ax2.plot(xs, [r["gap_pp"] for r in rows], color="#cc5c44", lw=1.8,
             label="gap = r_desired − target midpoint (pp)")
    ax2.plot(xs, [r["m_clicks"] for r in rows], color="#3d6b8e", lw=1.4, ls="--",
             label="m = per-meeting pressure (25bp clicks)")
    ax2.axhline(0, color="#8a8578", lw=0.8)
    ax2.set_ylabel("pp / clicks")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "newsagent_datachannel_dryrun.png", dpi=130)
    print(f"plot -> {PLOTS / 'newsagent_datachannel_dryrun.png'}")




# ---------------------------------------------------------------- scoring ------
def mpt_series(start: date, end: date, ref: str = "2026-06-17") -> list[dict]:
    """Atlanta Fed MPT — DISPLAY ONLY (DC-6). P(no change) over the 3-month SOFR
    window that CONTAINS the July meeting (reference_start 2026-06-17).

    CAVEAT, to be repeated wherever this is plotted: the window spans 2026-06-17 →
    2026-09-16, so 1 − P(cut) − P(hike) is a LOWER BOUND on P(no change at the July
    meeting alone) — a change anywhere in the window (including at the September
    meeting that closes it) counts against it. `Prob: cut` / `Prob: hike` are in
    PERCENT, not fractions.
    """
    import pandas as pd  # noqa: PLC0415
    f = CACHE / "mpt_histdata.xlsx"
    if not f.exists():
        f.write_bytes(_get(MPT_URL, 300))
    df = pd.read_excel(f, sheet_name="DATA")
    df["date"] = pd.to_datetime(df["date"])
    df["reference_start"] = pd.to_datetime(df["reference_start"])
    w = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
           & (df["reference_start"] == pd.Timestamp(ref))
           & (df["field"].isin(["Prob: cut", "Prob: hike"]))]
    out = []
    for d, g in w.groupby("date"):
        vals = dict(zip(g["field"], g["value"]))
        if {"Prob: cut", "Prob: hike"} <= set(vals):
            out.append({"date": d.date().isoformat(),
                        "p_hold_lb": round(1 - (vals["Prob: cut"] + vals["Prob: hike"]) / 100.0, 4),
                        "p_cut": round(vals["Prob: cut"] / 100.0, 4),
                        "p_hike": round(vals["Prob: hike"] / 100.0, 4),
                        "target_range": g["target_range"].iloc[0]})
    return sorted(out, key=lambda r: r["date"])


def news_fv() -> tuple[dict, float]:
    ser = json.loads((DATA / "fv_series.json").read_text()).get(SLUG, [])
    priors = json.loads((DATA / "priors.json").read_text())
    return ({p["date"]: p for p in ser}, priors[SLUG]["p0_pct"] / 100.0)


def brier(ps: list[float]) -> float:
    return float(np.mean([(p - OUTCOME) ** 2 for p in ps]))


def score(out: dict) -> dict:
    rows = out["rows"]
    by_date = {r["date"]: r for r in rows}
    fv, p0 = news_fv()
    inter = sorted(set(by_date) & set(fv))
    base = out["base_rate"]

    # (a) accuracy, on the intersection where a news FV exists
    a = {
        "n_intersection": len(inter),
        "first": inter[0] if inter else None, "last": inter[-1] if inter else None,
        "brier_p_struct": round(brier([by_date[d]["p_struct"] for d in inter]), 4),
        "brier_news_fv": round(brier([fv[d]["fv_pct"] / 100.0 for d in inter]), 4),
        "brier_prior_only": round(brier([p0] * len(inter)), 4),
        "brier_base_rate_only": round(brier([base] * len(inter)), 4),
        "brier_mid_context": round(brier([fv[d]["mid_pct"] / 100.0 for d in inter
                                          if fv[d].get("mid_pct") is not None]), 4),
        "brier_p_struct_full_window": round(brier([r["p_struct"] for r in rows]), 4),
        "n_full_window": len(rows),
    }
    a["a1_pass"] = a["brier_p_struct"] <= a["brier_news_fv"]
    a["a2_pass"] = a["brier_p_struct"] <= a["brier_prior_only"]

    # (b) directional sanity on mapped release days
    checks = []
    prev = None
    for r in rows:
        if prev is not None:
            d_pub_pce = r["last_pce_month"] != prev["last_pce_month"]
            d_sep = r["sep_rate"] != prev["sep_rate"]
            d_u = r["unrate"] != prev["unrate"]
            if d_pub_pce or d_sep or d_u:
                dp = r["p_struct"] - prev["p_struct"]
                kind = ("core-PCE print" if d_pub_pce else "") + (" SEP" if d_sep else "") \
                       + (" unemployment print" if d_u else "")
                if d_sep:
                    implied = "down" if r["sep_rate"] > prev["sep_rate"] else "up"
                elif d_pub_pce:
                    implied = "down" if r["pi_core_pce_yoy"] > prev["pi_core_pce_yoy"] else "up"
                else:
                    implied = None      # unemployment: no pre-registered direction
                viol = (implied == "down" and dp > 0.005) or (implied == "up" and dp < -0.005)
                checks.append({"date": r["date"], "kind": kind.strip(), "delta_p": round(dp, 4),
                               "pi_before": prev["pi_core_pce_yoy"], "pi_after": r["pi_core_pce_yoy"],
                               "sep_before": prev["sep_rate"], "sep_after": r["sep_rate"],
                               "implied_direction": implied, "violation": bool(viol)})
        prev = r
    b = {"checks": checks,
         "n_directional": sum(1 for c in checks if c["implied_direction"]),
         "violations": sum(1 for c in checks if c["violation"])}
    b["b_pass"] = b["violations"] == 0

    # (c) stability, in logits
    def lg(p): return math.log(p / (1 - p))
    steps = [abs(lg(rows[i]["p_struct"]) - lg(rows[i - 1]["p_struct"])) for i in range(1, len(rows))]
    c = {"max_daily_abs_dlogit": round(max(steps), 4),
         "max_daily_on": rows[1 + int(np.argmax(steps))]["date"],
         "total_abs_dlogit_end_to_end": round(abs(lg(rows[-1]["p_struct"]) - lg(rows[0]["p_struct"])), 4),
         "shift_cap_reference": 1.5}
    c["c_flag"] = c["max_daily_abs_dlogit"] > 1.5

    verdict = "GO" if (a["a1_pass"] and a["a2_pass"] and b["b_pass"] and not c["c_flag"]) else "NO-GO"
    return {"a": a, "b": b, "c": c, "verdict": verdict}

if __name__ == "__main__":
    sys.exit(main())
