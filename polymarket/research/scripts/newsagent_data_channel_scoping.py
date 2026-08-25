"""Free-consensus accuracy check for the Observatory's proposed data-evidence channel.

SCOPING ONLY — reads public data, writes CSVs, touches no model/dashboard/ledger code.
Pre-registration (metric + verdict rule locked BEFORE any comparison was computed) is
reproduced in notes/news_agent/newsagent_data_channel_scoping.md § Pre-registration.

Sources (all free, no paid provider):
  consensus/actual : Nasdaq economic calendar via OpenBB (`openbb-nasdaq`, keyless)
  official realized: FRED keyless CSV endpoint (fredgraph.csv) — no API key needed
  nowcast          : Cleveland Fed inflation-nowcasting public JSON (daily vintages)

OpenBB is NOT installed in the research venv (scoping pass — see the findings note).
Run it against a throwaway environment instead:

    uv run --python 3.14 \
      --with openbb-core --with openbb-economy --with openbb-nasdaq --with pandas \
      python scripts/newsagent_data_channel_scoping.py

Outputs (git-ignored, under data/):
  data/newsagent/datachannel/calendar_raw.csv     cached calendar pull
  data/analysis/csv_outputs/news_agent/newsagent_datachannel_consensus_pairs.csv
  data/analysis/csv_outputs/news_agent/newsagent_datachannel_consensus_summary.csv
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "newsagent" / "datachannel"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"
CACHE.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)

TODAY = date(2026, 8, 24)          # frozen run date (pre-registered window anchor)
START_EXT = date(2024, 8, 24)      # robustness-extension window start
START_PRI = date(2025, 8, 24)      # primary window start
N_PRIMARY = 12                     # most recent releases per series (primary sample)

# --- series spec (locked before metrics) -----------------------------------------
# kind: 'pct_mom' → FRED index → MoM %; 'level' → FRED level; 'diff_k' → FRED MoM diff (thousands)
SERIES = {
    "S1 CPI MoM":            dict(event="CPI",                  kind="pct_mom", fred="CPIAUCSL",  tol=0.05, dual=True),
    "S2 Core CPI MoM":       dict(event="Core CPI",             kind="pct_mom", fred="CPILFESL",  tol=0.05, dual=True),
    "S3 Nonfarm Payrolls":   dict(event="Nonfarm Payrolls",     kind="diff_k",  fred="PAYEMS",    tol=20.0, dual=False),
    "S4 Unemployment Rate":  dict(event="Unemployment Rate",    kind="level",   fred="UNRATE",    tol=0.05, dual=False),
    "S5 Core PCE MoM":       dict(event="Core PCE Price Index", kind="pct_mom", fred="PCEPILFE",  tol=0.05, dual=True),
    "S6 Retail Sales MoM":   dict(event="Retail Sales",         kind="pct_mom", fred="RSAFS",     tol=0.05, dual=True),
}
NOWCAST_SERIES = {"S1 CPI MoM": "CPI Inflation", "S2 Core CPI MoM": "Core CPI Inflation"}


# --- helpers ---------------------------------------------------------------------
def parse_val(s) -> float | None:
    """'0.2%' → 0.2 · '114K' → 114.0 (thousands) · '-1.30M' → -1300.0 · ''/'-' → None."""
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if t in ("", "-", "nan", "None"):
        return None
    mult = 1.0
    if t.endswith("%"):
        t = t[:-1]
    elif t.endswith("K"):
        t, mult = t[:-1], 1.0
    elif t.endswith("M"):
        t, mult = t[:-1], 1000.0
    elif t.endswith("B"):
        t, mult = t[:-1], 1_000_000.0
    try:
        return float(t) * mult
    except ValueError:
        return None


def ref_month(d: date) -> tuple[int, int]:
    """Reference period of a release = the previous calendar month (uniform for all six)."""
    return (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)


def fred_csv(series_id: str) -> pd.Series:
    """Keyless fredgraph.csv, cached to disk (the public endpoint times out intermittently)."""
    local = CACHE / f"fred_{series_id}.csv"
    if local.exists():
        df = pd.read_csv(local)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "epsilon-research/1.0"})
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    df = pd.read_csv(io.StringIO(r.read().decode()))
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise RuntimeError(f"FRED fetch failed for {series_id}: {last}")
        df.to_csv(local, index=False)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fred_official(spec: dict, s: pd.Series, ym: tuple[int, int]) -> tuple[float | None, float | None]:
    """Return (primary, yoy) official values for reference month ym. yoy=None when N/A."""
    ts = pd.Timestamp(year=ym[0], month=ym[1], day=1)
    prev = ts - pd.DateOffset(months=1)
    yago = ts - pd.DateOffset(months=12)
    if ts not in s.index:
        return None, None
    if spec["kind"] == "level":
        return round(float(s[ts]), 2), None
    if spec["kind"] == "diff_k":
        if prev not in s.index:
            return None, None
        return round(float(s[ts] - s[prev]), 1), None
    mom = round(100 * (s[ts] / s[prev] - 1), 2) if prev in s.index else None
    yoy = round(100 * (s[ts] / s[yago] - 1), 2) if yago in s.index else None
    return mom, yoy


def load_calendar(start: date, end: date) -> pd.DataFrame:
    """Nasdaq economic calendar, cached, with a VERIFIED per-date fetch status.

    Two gotchas found in this scoping run, both material enough to log:
      1. `openbb-nasdaq`'s calendar fetcher raises "No record found" for a whole range
         when a no-event date resolves first in its async gather, and its concurrency
         (~30 dates at once) trips api.nasdaq.com rate-limiting (HTTP 403). Under a
         bisecting wrapper both failure modes degrade to SILENTLY EMPTY dates — which
         would masquerade as missing releases in a coverage metric.
      2. So the bulk pull goes direct to the same public endpoint the provider uses,
         sequentially and paced, recording ok/empty/failed per date. The OpenBB path is
         verified separately on a sample (see verify_openbb_path()); a coverage number
         must never be contaminated by our own fetch reliability.
    """
    import time

    cache = CACHE / "calendar_raw.csv"
    status_path = CACHE / "calendar_fetch_status.csv"
    have, status = None, {}
    if cache.exists() and status_path.exists():
        have = pd.read_csv(cache, parse_dates=["date"])
        st = pd.read_csv(status_path)
        status = dict(zip(st["date"], st["status"]))

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    todo = [d for d in days if status.get(str(d)) not in ("ok", "empty")]
    if not todo:
        return have

    rows: list[dict] = []
    for i, d in enumerate(todo):
        url = f"https://api.nasdaq.com/api/calendar/economicevents?date={d}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json"})
        got = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    got = json.loads(r.read().decode())
                break
            except Exception:  # noqa: BLE001 — rate limit / transient
                time.sleep(2.5 * (attempt + 1))
        if got is None:
            status[str(d)] = "failed"
            continue
        data = (got.get("data") or {}).get("rows") or []
        status[str(d)] = "ok" if data else "empty"
        for item in data:
            rows.append({**{k: v for k, v in item.items() if k != "gmt"}, "date": str(d)})
        if i % 25 == 0:
            print(f"  calendar {d} ({i}/{len(todo)})", flush=True)
        time.sleep(0.35)

    df = pd.DataFrame(rows)
    if have is not None and len(have):
        df = pd.concat([have, df], ignore_index=True)
    df = df.drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(cache, index=False)
    pd.DataFrame([{"date": k, "status": v} for k, v in sorted(status.items())]).to_csv(status_path, index=False)
    n_fail = sum(1 for v in status.values() if v == "failed")
    print(f"  fetch status: {sum(v == 'ok' for v in status.values())} ok / "
          f"{sum(v == 'empty' for v in status.values())} empty / {n_fail} failed", flush=True)
    if n_fail:
        print("  WARNING: failed dates present — coverage metrics understate the feed", flush=True)
    return df


def verify_openbb_path(sample_start: str = "2026-08-12", sample_end: str = "2026-08-14") -> str:
    """Prove the library path works on a sample (the install check's end-to-end leg)."""
    try:
        from openbb import obb

        d = obb.economy.calendar(provider="nasdaq", start_date=sample_start, end_date=sample_end).to_df()
        return f"openbb path OK — {len(d)} rows {sample_start}..{sample_end}"
    except Exception as exc:  # noqa: BLE001
        return f"openbb path FAILED — {type(exc).__name__}: {str(exc)[:120]}"


def load_nowcasts() -> dict[tuple[int, int], list[tuple[date, str, float]]]:
    """Cleveland Fed public JSON → {(year, month): [(vintage_date, series, value_pct)]}."""
    url = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
    req = urllib.request.Request(url, headers={"User-Agent": "epsilon-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = json.loads(r.read().decode())
    out: dict[tuple[int, int], list] = {}
    for entry in blob:
        cap = entry["chart"].get("subcaption", "")
        m = re.match(r"^(\d{4})-(\d{1,2})$", cap.strip())
        if not m:
            continue
        ty, tm = int(m.group(1)), int(m.group(2))
        labels = [c.get("label", "") for c in entry["categories"][0]["category"]]
        rows = []
        for ds in entry.get("dataset", []):
            name = ds.get("seriesname", "")
            for i, pt in enumerate(ds.get("data", [])):
                v = pt.get("value")
                if v in (None, "") or i >= len(labels):
                    continue
                lm = re.match(r"^(\d{2})/(\d{2})$", labels[i])
                if not lm:
                    continue
                mm, dd = int(lm.group(1)), int(lm.group(2))
                yy = ty if mm == tm else (ty + 1 if mm < tm else ty)  # target month or the one after
                try:
                    rows.append((date(yy, mm, dd), name, float(v)))
                except ValueError:
                    continue
        out[(ty, tm)] = rows
    return out


# --- build the release table -----------------------------------------------------
def build_pairs(cal: pd.DataFrame, nowcasts: dict) -> pd.DataFrame:
    # the direct endpoint calls it `eventName`; the OpenBB model renames it to `event`
    cal = cal.rename(columns={"eventName": "event"})
    cal = cal[cal["country"] == "United States"].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    fred_cache: dict[str, pd.Series] = {}
    rows, drops = [], []

    for label, spec in SERIES.items():
        s = fred_cache.setdefault(spec["fred"], fred_csv(spec["fred"]))
        sub = cal[cal["event"].astype(str).str.strip().str.casefold() == spec["event"].casefold()]
        for d, grp in sub.groupby(sub["date"].dt.date):
            ym = ref_month(d)
            prev_ym = (ym[0], ym[1] - 1) if ym[1] > 1 else (ym[0] - 1, 12)
            off_mom, off_yoy = fred_official(spec, s, ym)
            prev_mom, prev_yoy = fred_official(spec, s, prev_ym)
            cand = []
            for _, r in grp.iterrows():
                cand.append(dict(actual=parse_val(r.get("actual")), consensus=parse_val(r.get("consensus")),
                                 previous=parse_val(r.get("previous"))))
            # --- MoM/YoY disambiguation (locked rule): match the row's `previous` field —
            # an already-published value — against FRED's prior-month MoM vs YoY. Never uses
            # this release's actual or consensus, so it cannot bias the surprise metrics.
            pick, how = None, ""
            if not spec["dual"]:
                pick, how = cand[0], "single-row"
            elif len(cand) == 1 and prev_mom is not None and prev_yoy is not None:
                # A dual-unit event with only ONE row published that day is NOT automatically
                # the MoM row (the Dec-2025 CPI release carried the YoY row alone). Validate it
                # against the prior month's official MoM before accepting; drop otherwise.
                c = cand[0]
                if c["previous"] is not None and abs(c["previous"] - prev_mom) <= 0.3 \
                        and abs(c["previous"] - prev_mom) < abs(c["previous"] - prev_yoy):
                    pick, how = c, "single-row-validated"
                else:
                    drops.append(dict(series=label, date=str(d),
                                      reason="single row failed MoM units validation (likely YoY)"))
                    continue
            elif prev_mom is not None and prev_yoy is not None:
                scored = [(abs(c["previous"] - prev_mom) if c["previous"] is not None else 9e9,
                           abs(c["previous"] - prev_yoy) if c["previous"] is not None else 9e9, c)
                          for c in cand]
                mom_like = [(dm, dy, c) for dm, dy, c in scored if dm <= 0.3 and dm < dy]
                if len(mom_like) == 1:
                    pick, how = mom_like[0][2], "previous-match"
                else:  # declared tie-break
                    small = [c for c in cand if c["actual"] is not None and abs(c["actual"]) < 1.5]
                    if len(small) == 1:
                        pick, how = small[0], "magnitude-tiebreak"
            if pick is None:
                drops.append(dict(series=label, date=str(d), reason="ambiguous MoM/YoY row"))
                continue
            rows.append(dict(
                series=label, release_date=str(d), ref_month=f"{ym[0]}-{ym[1]:02d}", pick_rule=how,
                consensus=pick["consensus"], actual_calendar=pick["actual"], previous=pick["previous"],
                official_fred=off_mom, prev_official=prev_mom,
                nowcast=nowcast_for(label, ym, d, nowcasts),
            ))
    if drops:
        pd.DataFrame(drops).to_csv(CSV_OUT / "newsagent_datachannel_dropped_rows.csv", index=False)
        print(f"  dropped (reported, not silent): {len(drops)} rows")
    return pd.DataFrame(rows)


def nowcast_for(label, ym, release_day, nowcasts) -> float | None:
    name = NOWCAST_SERIES.get(label)
    if not name or ym not in nowcasts:
        return None
    pre = [(dt, v) for dt, nm, v in nowcasts[ym] if nm == name and dt < release_day]
    return round(pre[-1][1], 4) if pre else None


# --- metrics ---------------------------------------------------------------------
def rmse(x: pd.Series) -> float | None:
    x = x.dropna()
    return round(float(((x ** 2).mean()) ** 0.5), 4) if len(x) else None


def expected_releases(spec: dict, start: date, end: date) -> int:
    """Reference months in the window for which FRED carries an official print — i.e. months a
    release actually happened (so shutdown-suspended months are not counted as feed gaps)."""
    s = fred_csv(spec["fred"])
    months = {(t.year, t.month) for t in s.index}
    n, cur = 0, date(start.year, start.month, 1)
    while cur <= end:
        ref = ref_month(cur)
        if ref in months:
            n += 1
        cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return n


def metrics(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    win_start = START_PRI if tag == "primary" else START_EXT
    out = []
    for label, spec in SERIES.items():
        d = df[df["series"] == label].sort_values("release_date")
        if tag == "primary":
            d = d.tail(N_PRIMARY)
        n = len(d)
        exp = min(expected_releases(spec, win_start, TODAY), N_PRIMARY if tag == "primary" else 99)
        # M1 per the pre-registration: share of EXPECTED releases carrying a consensus.
        cov = (d["consensus"].notna().sum() / exp) if exp else None
        integ = ((d["actual_calendar"] - d["official_fred"]).abs() <= spec["tol"]).mean() \
            if n and d["official_fred"].notna().any() else None
        surp = (d["actual_calendar"] - d["consensus"]).dropna()
        naive = (d["actual_calendar"] - d["previous"]).dropna()
        nc = d.dropna(subset=["nowcast", "actual_calendar"])
        out.append(dict(
            sample=tag, series=label, n_expected=exp, n_found=n, n_scored=len(surp),
            M1_coverage=round(float(cov), 3) if cov is not None else None,
            M2_integrity=round(float(integ), 3) if integ is not None else None,
            M3_bias=round(float(surp.mean()), 4) if len(surp) else None,
            M4_rmse_consensus=rmse(surp),
            M4_mae_consensus=round(float(surp.abs().mean()), 4) if len(surp) else None,
            M5_rmse_naive=rmse(naive),
            M6_rmse_nowcast=rmse(nc["actual_calendar"] - nc["nowcast"]) if len(nc) else None,
            M6_n=len(nc),
        ))
    m = pd.DataFrame(out)
    m["M3_pass"] = m.apply(lambda r: (abs(r.M3_bias) <= 0.25 * r.M4_rmse_consensus)
                           if r.M3_bias is not None and r.M4_rmse_consensus else None, axis=1)
    m["M5_pass"] = m.apply(lambda r: (r.M4_rmse_consensus < r.M5_rmse_naive)
                           if r.M4_rmse_consensus and r.M5_rmse_naive else None, axis=1)
    return m


def supplementary(cal: pd.DataFrame) -> pd.DataFrame:
    """SUPPLEMENTARY (not part of the pre-registered verdict): integrity on a series that
    is NEVER revised — the NSA CPI index level — which separates "the free feed printed a
    wrong number" from "the number was later revised". Also splits the M1 shortfall into
    releases ABSENT from the feed vs releases present but carrying no consensus.
    """
    cal = cal.rename(columns={"eventName": "event"})
    cal = cal[cal["country"] == "United States"].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    nsa = fred_csv("CPIAUCNS")
    rows = []
    sub = cal[cal["event"].astype(str).str.strip() == "CPI Index, n.s.a."]
    for d, grp in sub.groupby(sub["date"].dt.date):
        ym = ref_month(d)
        ts = pd.Timestamp(year=ym[0], month=ym[1], day=1)
        if ts not in nsa.index:
            continue
        a = parse_val(grp.iloc[0].get("actual"))
        if a is None:
            continue
        rows.append(dict(release=str(d), ref_month=f"{ym[0]}-{ym[1]:02d}", feed_actual=a,
                         fred_nsa=round(float(nsa[ts]), 3), diff=round(a - float(nsa[ts]), 3)))
    df = pd.DataFrame(rows)
    if len(df):
        df["match_0.05"] = df["diff"].abs() <= 0.05
        print(f"  never-revised integrity (CPI index NSA): {df['match_0.05'].mean():.3f} "
              f"on n={len(df)} releases; max |diff| = {df['diff'].abs().max()}")
    df.to_csv(CSV_OUT / "newsagent_datachannel_integrity_nsa.csv", index=False)
    return df


def main() -> int:
    print("0/4 openbb library path …", flush=True)
    print("   ", verify_openbb_path(), flush=True)
    print("1/4 calendar …", flush=True)
    cal = load_calendar(START_EXT, TODAY)
    print(f"    {len(cal)} rows")
    print("2/4 nowcasts …", flush=True)
    nowcasts = load_nowcasts()
    print(f"    {len(nowcasts)} target months")
    print("3/4 pairing …", flush=True)
    pairs = build_pairs(cal, nowcasts)
    pairs = pairs[pairs["release_date"] >= str(START_EXT)]
    pairs.to_csv(CSV_OUT / "newsagent_datachannel_consensus_pairs.csv", index=False)
    print(f"    {len(pairs)} release rows")
    print("4/4 metrics …", flush=True)
    m = pd.concat([metrics(pairs[pairs.release_date >= str(START_PRI)], "primary"),
                   metrics(pairs, "extension")], ignore_index=True)
    m.to_csv(CSV_OUT / "newsagent_datachannel_consensus_summary.csv", index=False)
    print("5/5 supplementary …", flush=True)
    supplementary(cal)
    # M1 shortfall split: absent-from-feed vs present-without-consensus
    split = []
    for label, spec in SERIES.items():
        d = pairs[(pairs.series == label) & (pairs.release_date >= str(START_PRI))]
        exp = min(expected_releases(spec, START_PRI, TODAY), N_PRIMARY)
        split.append(dict(series=label, expected=exp, found_in_feed=len(d),
                          of_which_with_consensus=int(d["consensus"].notna().sum()),
                          absent_from_feed=exp - len(d)))
    sp = pd.DataFrame(split)
    sp.to_csv(CSV_OUT / "newsagent_datachannel_coverage_split.csv", index=False)
    print(sp.to_string(index=False))
    pd.set_option("display.width", 220)
    print(m.to_string(index=False))
    return 0


def chart() -> None:
    """Render the two-panel summary figure (coverage + anchor RMSE).

    Needs matplotlib, which lives in the RESEARCH venv (dev group) — not in the
    throwaway OpenBB env. Run after main():
        PYTHONPATH=. uv run python scripts/newsagent_data_channel_scoping.py --chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    CSV = Path("data/analysis/csv_outputs/news_agent")
    OUT = Path("data/analysis/plots/news_agent"); OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(CSV/"newsagent_datachannel_consensus_summary.csv")
    sp = pd.read_csv(CSV/"newsagent_datachannel_coverage_split.csv")
    p = m[m["sample"]=="primary"].set_index("series")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    lbl = [s.split(" ",1)[1] for s in sp.series]
    y = range(len(sp))
    ax[0].barh(y, sp.expected, color="#d8d5cc", label="scheduled releases (FRED-confirmed)")
    ax[0].barh(y, sp.of_which_with_consensus, color="#cc5c44", label="present in free feed WITH consensus")
    ax[0].set_yticks(list(y)); ax[0].set_yticklabels(lbl, fontsize=9); ax[0].invert_yaxis()
    ax[0].axvline(0.8*12, ls="--", lw=1, color="#49413c")
    ax[0].text(0.8*12+0.12, 0.15, "0.80 pre-registered bar", fontsize=8, color="#49413c")
    ax[0].set_xlabel("releases in the 12-month primary window"); ax[0].set_title("A · Free-consensus COVERAGE (M1)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False); ax[0].set_xlim(0, 13)
    sub = ["S1 CPI MoM", "S2 Core CPI MoM", "S5 Core PCE MoM", "S6 Retail Sales MoM"]
    import numpy as np
    x = np.arange(len(sub)); w = 0.27
    ax[1].bar(x-w, [p.loc[s,"M4_rmse_consensus"] for s in sub], w, color="#cc5c44", label="consensus")
    ax[1].bar(x,   [p.loc[s,"M5_rmse_naive"] for s in sub], w, color="#b8b7ad", label="naive (previous print)")
    ax[1].bar(x+w, [p.loc[s,"M6_rmse_nowcast"] if pd.notna(p.loc[s,"M6_rmse_nowcast"]) else 0 for s in sub],
              w, color="#49413c", label="Cleveland Fed nowcast")
    ax[1].set_xticks(x); ax[1].set_xticklabels([s.split(" ",1)[1] for s in sub], fontsize=9)
    ax[1].set_ylabel("RMSE of the forecast error (pp of the print)")
    ax[1].set_title("B · How well each ANCHOR predicts the print (M4/M5/M6)", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].text(2-w, 0.005, "no nowcast\npublished", fontsize=7, ha="center", color="#49413c")
    ax[1].text(3-w, 0.005, "no nowcast\npublished", fontsize=7, ha="center", color="#49413c")
    fig.suptitle("Free objective-data channel — the free consensus is ACCURATE where it appears, but appears for only half the releases", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT/"newsagent_datachannel_consensus_check.png", dpi=140)
    print("saved", OUT/"newsagent_datachannel_consensus_check.png")


if __name__ == "__main__":
    if "--chart" in sys.argv:
        chart()
        sys.exit(0)
    sys.exit(main())
