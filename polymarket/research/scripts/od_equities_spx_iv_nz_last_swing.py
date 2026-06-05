"""SPX daily up/down implied-vol N(z) last-swing data audit.

This is intentionally not an options-chain or ML build. It checks whether the
requested rule-based fair-value pass can be run from clean historical inputs:
Cboe VIX/VIX9D direct CSVs, CME ES public settlements, and Polymarket historical
best-ask snapshots. If those ingredients do not overlap the SPX fill sample,
the script writes a blocker append instead of substituting mark-to-mid or
yfinance-like proxies.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_equities_spx_iv_nz_last_swing.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import markdown_table, number
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
EXTERNAL = ROOT / "data" / "external" / "spx_iv_last_swing"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
LIVE_CLOB = ROOT / "data" / "live_clob"
NOTES = ROOT / "notes" / "options_delta"
BRAIN_TODO = REPO / "brain" / "TODO.md"

MARKET_DETAIL = CSV_OUT / "od_equities_index_pricing_scope_market_detail.csv"
RAW_FILLS = ANALYSIS / "od_equities_spx_nz_pricing_raw_fills.parquet"
NOTE = NOTES / "od_equities_index_pricing_scope_findings.md"
HUB = NOTES / "strat_options_delta.md"

OUT_AUDIT = CSV_OUT / "od_equities_spx_iv_nz_last_swing_data_audit.csv"
OUT_CME = CSV_OUT / "od_equities_spx_iv_nz_last_swing_cme_es_probe.csv"
OUT_VIX = CSV_OUT / "od_equities_spx_iv_nz_last_swing_vix_coverage.csv"
OUT_CLOB = CSV_OUT / "od_equities_spx_iv_nz_last_swing_clob_scan.csv"

CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CBOE_VIX9D_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"
CME_ES_SETTLEMENTS_PAGE = "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.settlements.html"
CME_ES_SETTLEMENTS_API = "https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/133/FUT?tradeDate={trade_date}"

HTTP_HEADERS = {
    "User-Agent": "epsilon-quant-research-spx-iv-last-swing/1.0 Mozilla/5.0",
    "Accept": "application/json,text/csv,text/html",
}


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    try:
        out = json.loads(str(value))
        return out if isinstance(out, list) else []
    except Exception:
        return []


def event_date_from_slug(slug: str) -> pd.Timestamp:
    m = re.search(r"on-([a-z]+)-(\d{1,2})-(\d{4})", str(slug))
    if not m:
        return pd.NaT
    month, day, year = m.groups()
    return pd.Timestamp(f"{year}-{month}-{int(day):02d}", tz="UTC")


def clean_number(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).replace(",", "").strip()
    text = re.sub(r"[ABCDEF]+$", "", text)
    try:
        return float(text)
    except Exception:
        return math.nan


def spx_markets() -> pd.DataFrame:
    df = pd.read_csv(MARKET_DETAIL)
    df = df[(df["family"].eq("index_up_down")) & (df["underlying"].eq("SPX"))].copy()
    df["market_id"] = df["market_id"].astype(str)
    df["event_date"] = df["market_slug"].map(event_date_from_slug)
    df["token_list"] = df["clob_token_ids"].map(parse_json_list)
    df = df[df["event_date"].notna() & df["token_list"].map(len).ge(2)].copy()
    return df.sort_values("event_date").reset_index(drop=True)


def spx_fill_dates() -> pd.DataFrame:
    fills = pd.read_parquet(RAW_FILLS)
    fills["event_date"] = pd.to_datetime(fills["event_date"], utc=True)
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    return (
        fills.groupby("event_date", as_index=False)
        .agg(
            fill_rows=("timestamp", "size"),
            first_fill_ts=("timestamp", "min"),
            last_fill_ts=("timestamp", "max"),
            markets=("market_id", "nunique"),
        )
        .sort_values("event_date")
    )


def fetch_cboe_csv(name: str, url: str, refresh: bool = False) -> pd.DataFrame:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / f"{name.lower()}_history.csv"
    if not path.exists() or refresh:
        with httpx.Client(headers=HTTP_HEADERS, timeout=30, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        time.sleep(0.25)
    out = pd.read_csv(path)
    out["event_date"] = pd.to_datetime(out["DATE"], format="%m/%d/%Y", utc=True)
    out[f"{name.lower()}_close"] = out["CLOSE"].astype(float)
    out["source_url"] = url
    return out[["event_date", f"{name.lower()}_close", "source_url"]].copy()


def fetch_cme_es_probe(dates: list[pd.Timestamp], refresh: bool = False) -> pd.DataFrame:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / "cme_es_settlements_probe.csv"
    if path.exists() and not refresh:
        out = pd.read_csv(path)
        out["event_date"] = pd.to_datetime(out["event_date"], utc=True)
        return out

    rows: list[dict[str, Any]] = []
    with httpx.Client(headers={**HTTP_HEADERS, "Referer": CME_ES_SETTLEMENTS_PAGE}, timeout=20, follow_redirects=True) as client:
        for ts in dates:
            trade_date = ts.strftime("%m/%d/%Y")
            url = CME_ES_SETTLEMENTS_API.format(trade_date=trade_date)
            try:
                response = client.get(url)
                status = response.status_code
                payload = response.json() if response.content else {}
            except Exception as exc:
                rows.append(
                    {
                        "event_date": ts,
                        "trade_date": trade_date,
                        "status_code": math.nan,
                        "has_settlement": False,
                        "front_month": "",
                        "front_settle": math.nan,
                        "rows": 0,
                        "source_url": url,
                        "error": str(exc),
                    }
                )
                continue
            settlements = payload.get("settlements") or []
            front = settlements[0] if settlements else {}
            rows.append(
                {
                    "event_date": ts,
                    "trade_date": trade_date,
                    "status_code": status,
                    "has_settlement": bool(settlements),
                    "front_month": front.get("month", ""),
                    "front_settle": clean_number(front.get("settle")),
                    "rows": len(settlements),
                    "source_url": url,
                    "error": "",
                }
            )
            time.sleep(0.05)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def scan_local_spx_clob(markets: pd.DataFrame) -> pd.DataFrame:
    slugs = sorted(set(markets["market_slug"].astype(str)))
    patterns = ["spx-up-or-down", *slugs]
    pattern_path = EXTERNAL / "spx_daily_slug_patterns.txt"
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    pattern_path.write_text("\n".join(patterns) + "\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            ["rg", "-n", "-i", "-F", "-f", str(pattern_path), str(LIVE_CLOB)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "path": "rg_scan",
                    "matched_lines": 0,
                    "book_lines": 0,
                    "best_bid_ask_lines": 0,
                    "error": str(exc),
                }
            ]
        )
    if proc.returncode not in {0, 1}:
        return pd.DataFrame(
            [
                {
                    "path": "rg_scan",
                    "matched_lines": 0,
                    "book_lines": 0,
                    "best_bid_ask_lines": 0,
                    "error": proc.stderr.strip(),
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for raw_line in proc.stdout.splitlines():
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        path, _, line = parts
        row = by_path.setdefault(
            path,
            {
                "path": str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path,
                "matched_lines": 0,
                "book_lines": 0,
                "best_bid_ask_lines": 0,
                "error": "",
            },
        )
        row["matched_lines"] += 1
        if '"event_type":"book"' in line or '"event_type": "book"' in line:
            row["book_lines"] += 1
        if "best_bid_ask" in line:
            row["best_bid_ask_lines"] += 1
    rows = list(by_path.values())
    return pd.DataFrame(rows, columns=["path", "matched_lines", "book_lines", "best_bid_ask_lines", "error"])


def pct_ratio(n: int, d: int) -> str:
    return "n/a" if d <= 0 else f"{n:,}/{d:,} ({n / d:.1%})"


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    rows = []
    for _, row in df.iterrows():
        out = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                out.append(number(float(value), 3) if np.isfinite(value) else "n/a")
            elif isinstance(value, (bool, np.bool_)):
                out.append("true" if bool(value) else "false")
            elif isinstance(value, (int, np.integer)):
                out.append(f"{int(value):,}")
            else:
                out.append(str(value))
        rows.append(out)
    return markdown_table(cols, rows)


def write_outputs(markets: pd.DataFrame, fills: pd.DataFrame, vix: pd.DataFrame, vix9d: pd.DataFrame, cme: pd.DataFrame, clob: pd.DataFrame) -> pd.DataFrame:
    fill_dates = fills["event_date"].drop_duplicates().sort_values()
    raw_fill_rows = int(fills["fill_rows"].sum())
    vix_fill = pd.DataFrame({"event_date": fill_dates}).merge(vix, on="event_date", how="left")
    vix_fill = vix_fill.merge(vix9d, on="event_date", how="left")
    cme_fill = pd.DataFrame({"event_date": fill_dates}).merge(cme, on="event_date", how="left")
    cme_scope = markets[["event_date"]].drop_duplicates().merge(cme, on="event_date", how="left")

    clob_book_lines = int(clob["book_lines"].sum()) if not clob.empty else 0
    clob_best_lines = int(clob["best_bid_ask_lines"].sum()) if not clob.empty else 0
    cme_fill_hits = int(cme_fill["has_settlement"].fillna(False).sum())
    cme_scope_hits = int(cme_scope["has_settlement"].fillna(False).sum())
    vix_hits = int(vix_fill["vix_close"].notna().sum())
    vix9d_hits = int(vix_fill["vix9d_close"].notna().sum())

    audit = pd.DataFrame(
        [
            {
                "ingredient": "Cboe VIX close",
                "required_for_strict_gate": True,
                "available": vix_hits == len(fill_dates),
                "coverage": pct_ratio(vix_hits, len(fill_dates)),
                "detail": "Direct Cboe VIX CSV covers every local SPX fill market-date.",
            },
            {
                "ingredient": "Cboe VIX9D close",
                "required_for_strict_gate": False,
                "available": vix9d_hits == len(fill_dates),
                "coverage": pct_ratio(vix9d_hits, len(fill_dates)),
                "detail": "Direct Cboe VIX9D CSV is available as a short-horizon diagnostic, but the user-requested primary input is VIX.",
            },
            {
                "ingredient": "CME ES front-month settlement",
                "required_for_strict_gate": True,
                "available": cme_fill_hits == len(fill_dates),
                "coverage": f"fill sample {pct_ratio(cme_fill_hits, len(fill_dates))}; scope {pct_ratio(cme_scope_hits, markets['event_date'].nunique())}",
                "detail": "CME public settlement endpoint only returns recent May 27-June 2 rows. The single fill-date overlap is May 27, whose local PM rows stop before that day's resolution close.",
            },
            {
                "ingredient": "Historical PM SPX daily up/down best ask",
                "required_for_strict_gate": True,
                "available": clob_book_lines > 0 or clob_best_lines > 0,
                "coverage": f"{clob_book_lines:,} book lines / {clob_best_lines:,} best-bid-ask lines matched",
                "detail": "Local live_clob files have no SPX daily up/down book snapshots. Existing SPX historical sample is actual fills only.",
            },
            {
                "ingredient": "Local PM SPX daily up/down fills",
                "required_for_strict_gate": False,
                "available": True,
                "coverage": f"{raw_fill_rows:,} fill rows across {len(fill_dates):,} market-dates",
                "detail": "Useful for the earlier no-mid fill-level N(z) close, but not a literal best-ask replay.",
            },
        ]
    )
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_AUDIT, index=False)
    cme.to_csv(OUT_CME, index=False)
    vix_fill.to_csv(OUT_VIX, index=False)
    clob.to_csv(OUT_CLOB, index=False)
    return audit


def append_note(audit: pd.DataFrame, markets: pd.DataFrame, fills: pd.DataFrame, cme: pd.DataFrame) -> None:
    fill_start = fills["event_date"].min().strftime("%Y-%m-%d")
    fill_end = fills["event_date"].max().strftime("%Y-%m-%d")
    fill_rows = int(fills["fill_rows"].sum())
    fill_dates = fills["event_date"].nunique()
    last_fill_ts = pd.to_datetime(fills["last_fill_ts"], utc=True).max().strftime("%Y-%m-%d %H:%M:%S UTC")
    cme_hits = int(cme["has_settlement"].fillna(False).sum())
    cme_dates = cme[cme["has_settlement"].fillna(False)]["event_date"].dt.strftime("%Y-%m-%d").tolist()

    section = f"""
## 2026-06-03 Implied-Vol N(z) Last Swing Audit

Strict verdict: **STILL-BLOCKED for the exact VIX + ES + best-ask gate**. This does **not** create a MERITS-BUILD signal and does **not** justify an options-chain or ML build. The broader SPX pricing thesis remains **CONFIRM-CLOSE** on the completed realized-vol/fill-level gate above, but the requested last swing cannot be reconstructed from available clean historical inputs.

What was requested: replace realized volatility with forward-looking **Cboe VIX** and replace cash spot/drift with **ES futures**, then compare to executable **Polymarket best ask** with market-date clustered CI. I did not substitute yfinance, midpoint marks, last-trade marks, or an options chain.

### Data Audit

{md_table(audit, ['ingredient', 'required_for_strict_gate', 'available', 'coverage', 'detail'])}

Read: the Cboe vol input is not the blocker. Direct Cboe VIX/VIX9D files cover the local SPX sample. The blockers are the two executable/history legs: local PM data has **{fill_rows:,}** SPX daily up/down fill rows across **{fill_dates:,}** market-dates ({fill_start} to {fill_end}), but no historical SPX daily up/down CLOB best-ask snapshots; the local PM tape's latest row is **{last_fill_ts}**. CME's public ES settlement endpoint returned **{cme_hits:,}** scope-date hits ({', '.join(cme_dates) if cme_dates else 'none'}). The only raw fill-date overlap is May 27, but those PM rows occur before the local tape ends on May 26, not near the May 27 resolution close required for the requested best-ask comparison.

### Why Not a Proxy

Using actual fills again would answer a different question: "did observed fills look rich/cheap versus a VIX-based model?" That is useful only as a sensitivity, and it would still need an ES-forward input for moneyness. Using the old Yahoo cash states would violate this task's clean-source instruction. Using PM last trade or fill VWAP as "best ask" would reintroduce the same non-executable quote problem the prompt explicitly ruled out.

### Missing To Run The Strict Gate

- Historical Polymarket CLOB snapshots for SPX daily up/down token best ask/best bid, with timestamps before the official SPX close.
- Historical ES front-month prices aligned to those PM quote timestamps, or at minimum CME settlement/last data for the same historical SPX market dates; the public CME endpoint checked here only exposed recent rows after the local PM fill tape ended.
- Official SPX settlement remains available through the resolved PM markets / prior close logic; that is not the blocker.

### Decision

**No ML, no options-chain build, no OPRA spend.** The rule-based last swing cannot be made decision-grade from the current cache without violating the no-mark-to-mid/no-yfinance discipline. If this is ever reopened, the next step is a live SPX daily up/down collector that logs PM best ask/bid plus Cboe VIX and ES front-month state at quote time; until then, the pricing branch stays closed by the realized-vol gate and blocked for the stricter VIX+ES replay.

Outputs:
- Script: `scripts/od_equities_spx_iv_nz_last_swing.py`
- Audit: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_data_audit.csv`
- VIX coverage: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_vix_coverage.csv`
- CME ES probe: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_cme_es_probe.csv`
- Local CLOB scan: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_clob_scan.csv`

References checked:
- [Cboe VIX historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data)
- [Cboe direct VIX CSV](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)
- [CME E-mini S&P 500 settlements](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.settlements.html)
"""
    text = NOTE.read_text(encoding="utf-8")
    marker = "## 2026-06-03 Implied-Vol N(z) Last Swing Audit"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    text = re.sub(
        r"status: capacity-pass-pricing-confirm-close-no-opra-build(?:-iv-swing-blocked)*",
        "status: capacity-pass-pricing-confirm-close-no-opra-build-iv-swing-blocked",
        text,
    )
    NOTE.write_text(normalize_markdown_wrapping(text), encoding="utf-8")


def update_docs() -> None:
    hub_line = "- 2026-06-03 SPX implied-vol N(z) last swing: **STILL-BLOCKED as strict VIX+ES+best-ask replay; no ML/options build**. Cboe VIX/VIX9D direct data covers the SPX fill sample, but local PM data has fills rather than historical SPX daily up/down best asks. CME's public ES settlement endpoint only has one raw fill-date overlap, and that PM data stops before the May 27 resolution close, so there is no decision-grade best-ask/ES overlap. Pricing branch remains CONFIRM-CLOSE from the realized-vol gate; no fresh executable residual. See [[od_equities_index_pricing_scope_findings]]."
    hub = HUB.read_text(encoding="utf-8")
    if "SPX implied-vol N(z) last swing" in hub:
        hub = re.sub(
            r"- 2026-06-03 SPX implied-vol N\(z\) last swing: .+\n",
            hub_line + "\n",
            hub,
            count=1,
        )
    else:
        marker = "## Current state (2026-06-03)"
        insert = hub.find("\n-", hub.find(marker))
        if insert >= 0:
            hub = hub[:insert] + "\n" + hub_line + hub[insert:]
    hub = hub.replace(
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE, no Cboe/OPRA build until a future cheap realized-vol residual survives. See [[od_equities_index_pricing_scope_findings]].",
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE. Follow-on VIX+ES last-swing audit is strict-data-blocked, so no Cboe/OPRA/ML build. See [[od_equities_index_pricing_scope_findings]].",
    )
    HUB.write_text(normalize_markdown_wrapping(hub), encoding="utf-8")

    todo_line = "- [x] **SPX implied-vol N(z) last swing audit** (2026-06-03): completed; strict VIX+ES+best-ask replay is data-blocked. Cboe VIX/VIX9D direct data covers the local sample, but historical PM SPX daily best asks and overlapping historical CME ES prices are missing; no ML/options-chain build. See [[od_equities_index_pricing_scope_findings]]."
    todo_state_line = "- 2026-06-03 OD SPX implied-vol N(z) last swing: **STILL-BLOCKED for strict VIX+ES+best-ask replay; no ML/options build**. Cboe VIX/VIX9D direct data covers the local sample, but historical PM SPX daily best asks and resolution-time overlapping CME ES prices are missing. Pricing branch remains CONFIRM-CLOSE from the realized-vol gate. See [[od_equities_index_pricing_scope_findings]]."
    todo = BRAIN_TODO.read_text(encoding="utf-8")
    if "OD SPX implied-vol N(z) last swing" in todo:
        todo = re.sub(
            r"- 2026-06-03 OD SPX implied-vol N\(z\) last swing: .+\n",
            todo_state_line + "\n",
            todo,
            count=1,
        )
    else:
        marker = "## OD — Options-Delta"
        insert = todo.find("\n-", todo.find(marker))
        if insert >= 0:
            todo = todo[:insert] + "\n" + todo_state_line + todo[insert:]
    if "SPX implied-vol N(z) last swing audit" not in todo:
        anchor = "- [x] **SPX daily up/down N(z)/realized-vol pricing gate**"
        idx = todo.find(anchor)
        if idx >= 0:
            end = todo.find("\n", idx)
            todo = todo[: end + 1] + todo_line + "\n" + todo[end + 1 :]
    todo = todo.replace(
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE. Do not build Cboe/OPRA unless a future cheap realized-vol residual first clears lower-CI-positive net executable edge. See [[od_equities_index_pricing_scope_findings]].",
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE. Follow-on VIX+ES last-swing audit is strict-data-blocked, so do not build Cboe/OPRA/ML unless a future live best-ask collector first produces a lower-CI-positive net executable edge. See [[od_equities_index_pricing_scope_findings]].",
    )
    BRAIN_TODO.write_text(normalize_markdown_wrapping(todo), encoding="utf-8")


def main() -> None:
    markets = spx_markets()
    fills = spx_fill_dates()
    all_dates = sorted(markets["event_date"].drop_duplicates().tolist())
    vix = fetch_cboe_csv("VIX", CBOE_VIX_URL)
    vix9d = fetch_cboe_csv("VIX9D", CBOE_VIX9D_URL)
    cme = fetch_cme_es_probe(all_dates)
    clob = scan_local_spx_clob(markets)
    audit = write_outputs(markets, fills, vix, vix9d, cme, clob)
    append_note(audit, markets, fills, cme)
    update_docs()
    print(audit.to_string(index=False), flush=True)
    print(f"wrote {OUT_AUDIT}", flush=True)
    print(f"wrote {NOTE}", flush=True)


if __name__ == "__main__":
    main()
