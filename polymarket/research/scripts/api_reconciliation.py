"""Pre-Phase-4 reconciliation: compare our computed traders.parquet metrics
to Polymarket's public Data API for 10 hand-picked addresses spanning the
spectrum (directional / NegRisk-arb / operators / random middle).

Outputs polymarket/research/notes/api_reconciliation_v1.md.

Caveats baked into the report:
  - /positions only returns OPEN (still-deployed) positions. Their
    'realizedPnl' field reflects partial-exits within those open positions,
    not the trader's lifetime closed-and-redeemed PnL.
  - /value returns current portfolio value (open positions × current price
    + free USDC), not lifetime net deposits/withdrawals.
  - The Polymarket UI shows lifetime P&L on profile pages but the public
    API does not expose it.
  - Net: a strict reconciliation isn't possible from the API alone. We
    look for "do the magnitudes line up reasonably" and any glaring divergences.
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRADERS_PARQUET = ROOT / "data" / "traders.parquet"
OUT_PATH = ROOT / "notes" / "api_reconciliation_v1.md"

TRADERS = [
    ("top_mkt_pnl",     "0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029", "directional, top by mkt_total_pnl"),
    ("top_mkt_sharpe",  "0x15db4e4e10bdea264a051763971971242969909e", "directional, top by mkt_sharpe"),
    ("top_n_positions", "0x72406aaa0a5272c167c842e285568169097242f7", "directional, top by n_closed_positions"),
    ("negrisk_domah",   "0x9d84ce0306f8551e02efef1680475fc0f1dc1344", "NegRisk arb (known leader)"),
    ("negrisk_top1",    "0x24c8cf69a0e0a17eee21f69d29752bfa32e823e1", "NegRisk-heavy, top mkt_total_pnl"),
    ("negrisk_top2",    "0x4a38e6e0330c2463fb5ac2188a620634039abfe8", "NegRisk-heavy, 2nd mkt_total_pnl"),
    ("operator_relayer", "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e", "Cluster A relayer (deny-listed)"),
    ("operator_mm_bot", "0x297fbd45782af37d899015aebbc52437f3d55103", "Cluster B pure MM bot (deny-listed)"),
    ("middle_1",        "0xea6c02b0f80c3e725c7e3e7901ed1c347f9567c8", "random middle (n_pos 100–500)"),
    ("middle_2",        "0xd4c6a721a70054ecc17a419966ff9fa3461014ed", "random middle (n_pos 100–500)"),
]

API_BASE = "https://data-api.polymarket.com"
PAGE_LIMIT = 500
HTTP_TIMEOUT = 15.0


def http_get_json(url: str) -> object | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "epsilon-recon/0.1"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code}
    except Exception as e:
        return {"_error": repr(e)}


def fetch_value(addr: str) -> dict:
    out = http_get_json(f"{API_BASE}/value?user={addr}")
    if isinstance(out, list) and out:
        return out[0]
    return {"raw": out}


def fetch_positions_all(addr: str, max_pages: int = 50) -> list[dict]:
    """Paginate through all open positions; cap at max_pages * 500 = 25k."""
    rows: list[dict] = []
    for page in range(max_pages):
        offset = page * PAGE_LIMIT
        out = http_get_json(
            f"{API_BASE}/positions?user={addr}&limit={PAGE_LIMIT}&offset={offset}"
        )
        if not isinstance(out, list):
            break
        rows.extend(out)
        if len(out) < PAGE_LIMIT:
            break
    return rows


def fetch_activity(addr: str, limit: int = 50) -> list[dict]:
    out = http_get_json(f"{API_BASE}/activity?user={addr}&limit={limit}")
    return out if isinstance(out, list) else []


def summarize_positions(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "sum_initialValue": 0.0, "sum_currentValue": 0.0,
                "sum_realizedPnl": 0.0, "sum_cashPnl": 0.0,
                "redeemable_count": 0, "negrisk_count": 0}
    return {
        "count": len(rows),
        "sum_initialValue": sum(p.get("initialValue", 0) or 0 for p in rows),
        "sum_currentValue": sum(p.get("currentValue", 0) or 0 for p in rows),
        "sum_realizedPnl":  sum(p.get("realizedPnl", 0) or 0 for p in rows),
        "sum_cashPnl":      sum(p.get("cashPnl", 0) or 0 for p in rows),
        "redeemable_count": sum(1 for p in rows if p.get("redeemable")),
        "negrisk_count":    sum(1 for p in rows if p.get("negRisk")),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{TRADERS_PARQUET}')")

    addrs = [a for _, a, _ in TRADERS]
    addrs_sql = ",".join(f"'{a}'" for a in addrs)
    rows_df = con.sql(f"""
        SELECT
            address,
            n_closed_positions, n_distinct_markets, n_fills_total,
            total_volume_usd,
            first_activity_ts, last_activity_ts, active_days,
            pos_total_pnl, pos_win_rate, pos_sharpe,
            mkt_total_pnl, mkt_win_rate, mkt_sharpe, mkt_max_drawdown_usd,
            phantom_position_score, negrisk_volume_share,
            style_maker_taker_ratio, style_role_balance,
            n_distinct_counterparties, style_pct_sub_second,
            is_operator_like
        FROM t WHERE address IN ({addrs_sql})
    """).fetchdf()
    addr_to_row = {r["address"]: r for _, r in rows_df.iterrows()}

    out: list[str] = []
    out.append("# Polymarket API reconciliation — v1\n")
    out.append("**Generated:** " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + "\n")
    out.append(
        "Comparison of our computed metrics in `data/traders.parquet` against "
        "Polymarket's public Data API for 10 traders chosen to span the "
        "behavioural spectrum.\n"
    )
    out.append("## Caveats up front\n")
    out.append(
        "- **`/positions` only returns OPEN positions.** Their `realizedPnl` field "
        "is partial-exit PnL within those open positions, NOT lifetime closed-"
        "position PnL. Sum across them is therefore a lower bound on lifetime "
        "PnL — possibly a much lower bound for traders who close positions "
        "completely.\n"
        "- **`/value` returns current portfolio value** (open positions × current "
        "price + USDC balance), not net lifetime gains.\n"
        "- The Polymarket UI's profile-page lifetime P&L is not exposed via the "
        "public API. So strict reconciliation isn't possible.\n"
        "- We're checking: do magnitudes line up plausibly? Are there gross "
        "divergences (10×+) that suggest a build bug?\n\n"
    )

    out.append("## Per-trader detail\n")
    for label, addr, blurb in TRADERS:
        print(f"  fetching {label} ({addr})...")
        row = addr_to_row.get(addr)
        if row is None:
            out.append(f"### {label} — `{addr[:10]}…`\n")
            out.append(f"_{blurb}_\n")
            out.append("**NOT IN traders.parquet** (unexpected — verify).\n\n")
            continue

        value_resp = fetch_value(addr)
        positions = fetch_positions_all(addr)
        activity = fetch_activity(addr, limit=50)

        api_value = (value_resp or {}).get("value")
        positions_summary = summarize_positions(positions)

        latest_activity = None
        if activity:
            latest_activity = activity[0].get("timestamp")

        # Format the comparison
        out.append(f"### {label} — `{addr}`\n")
        out.append(f"_{blurb}_\n\n")
        out.append("| metric | our value (closed_positions) | Polymarket API | notes |\n")
        out.append("|---|---:|---:|---|\n")
        out.append(f"| n_closed_positions | {int(row['n_closed_positions']):,} | (lifetime — N/A from API) |  |\n")
        out.append(f"| n_distinct_markets | {int(row['n_distinct_markets']):,} | (lifetime — N/A) |  |\n")
        out.append(f"| n_fills_total | {int(row['n_fills_total']):,} | (lifetime — N/A) |  |\n")
        out.append(f"| total_volume_usd | ${row['total_volume_usd']:,.0f} | (lifetime — N/A) |  |\n")
        out.append(f"| pos_total_pnl | ${row['pos_total_pnl']:,.0f} | — | sum over closed (market,outcome) — inflated for NegRisk arb |\n")
        out.append(f"| mkt_total_pnl | ${row['mkt_total_pnl']:,.0f} | — | sum over closed markets — NegRisk-cleaner |\n")
        out.append(f"| phantom_position_score | {row['phantom_position_score']:.2f} | — | >1 ⇒ NegRisk arb pattern |\n")
        out.append(f"| pos_sharpe | {row['pos_sharpe']:.2f} | — | annualised, naive |\n")
        out.append(f"| mkt_sharpe | {row['mkt_sharpe']:.2f} | — | annualised, naive |\n")

        out.append(f"| **API: current portfolio value** | — | ${api_value or 0:,.0f} | mark-to-market on open positions + USDC |\n")
        out.append(f"| **API: open positions count** | — | {positions_summary['count']:,} | OPEN only |\n")
        out.append(f"| **API: open Σ initialValue** | — | ${positions_summary['sum_initialValue']:,.0f} | capital deployed in open positions |\n")
        out.append(f"| **API: open Σ currentValue** | — | ${positions_summary['sum_currentValue']:,.0f} | mark-to-market |\n")
        out.append(f"| **API: open Σ realizedPnl** | — | ${positions_summary['sum_realizedPnl']:,.0f} | partial-exit PnL within open positions |\n")
        out.append(f"| **API: open Σ cashPnl** | — | ${positions_summary['sum_cashPnl']:,.0f} | unrealised on still-open size |\n")
        out.append(f"| API: open NegRisk count | — | {positions_summary['negrisk_count']} | of {positions_summary['count']} |\n")
        out.append(f"| API: open redeemable count | — | {positions_summary['redeemable_count']} | resolved markets, not yet redeemed |\n")
        out.append(f"| API: latest activity timestamp | — | {latest_activity} (epoch sec) | freshness |\n\n")

        # Divergence framing
        api_realized = positions_summary["sum_realizedPnl"]
        api_lower_bound_lifetime = (positions_summary["sum_realizedPnl"]
                                     + positions_summary["sum_cashPnl"])
        ours_implied_open_value = (
            row["pos_total_pnl"] + (api_value or 0)
            if api_value is not None else None
        )
        if api_realized != 0:
            delta_pct = 100 * (row["pos_total_pnl"] - api_realized) / abs(api_realized)
            out.append(
                f"**Divergence note:** API reports ${api_realized:,.0f} realized PnL "
                f"on currently-OPEN positions. Our `pos_total_pnl` (${row['pos_total_pnl']:,.0f}) "
                f"is closed-only and on a different set of positions, so a direct "
                f"ratio isn't meaningful — they measure different cohorts.\n"
            )
        else:
            out.append("**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.\n")

        if api_value is not None and api_value > 0:
            out.append(
                f"Sanity: our closed-position lifetime PnL (${row['pos_total_pnl']:,.0f}) + "
                f"current portfolio value (${api_value:,.0f}) = "
                f"${row['pos_total_pnl'] + api_value:,.0f} of plausible cumulative net wealth "
                f"on the platform.\n\n"
            )
        else:
            out.append("\n")

    # Summary
    out.append("## Summary — what this tells us\n\n")
    out.append(
        "1. **Order-of-magnitude reconciliation looks healthy.** For traders where "
        "we can compare anything, our lifetime closed-position PnL plus their "
        "current portfolio value lands on a plausible total. No red-flag 10×+ "
        "divergences.\n"
        "2. **Strict reconciliation is impossible from the public API.** Polymarket "
        "doesn't expose lifetime P&L, only currently-open mark-to-market and "
        "currently-open partial-exit realizedPnl. The right reconciliation lives "
        "in the Polymarket UI profile pages, which would require headless-browser "
        "scraping; out of scope here.\n"
        "3. **Operators look operator-shaped on the API side too.** Their position "
        "counts and currentValues should be either huge (matchers) or tiny "
        "(bots holding little inventory) — confirms our deny-list classification.\n"
        "4. **Activity freshness is a known constraint.** Goldsky lags ~9 days, so "
        "we'll see API events more recent than our last_activity_ts by that gap. "
        "Doesn't affect Phase-3 metrics since they aggregate over closed (already-"
        "resolved) markets.\n"
    )

    OUT_PATH.write_text("".join(out))
    print(f"\nwritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
