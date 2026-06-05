"""Smoke-test views.sql before relying on it for the closed-positions build.

Designed to be fast (~30 s total) by using sargable single-value WHERE filters
that push down through the UNION ALL inside trader_actions. Avoids:
  - count(*) over the full 2 B-row exploded view (DuckDB materialises it)
  - WHERE x IN (subquery) on the view (predicate doesn't push through UNION ALL)

Tests:
  1. Row counts (raw_trades, joined_fills, markets_tokens, orphan)
  2. Domah's stats via trader_actions match the validation report
  3. Sign symmetry: for 5 specific transaction hashes, sum(token_delta) per
     (transaction_hash, outcome_token_id) is 0 within float tolerance

Halt with non-zero exit if any check fails.
"""
import sys
import time

from data_infra.duck import connect
from data_infra.views import load_views

DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"


def timed(label: str):
    return TimedSection(label)


class TimedSection:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        print(f"\n--- {self.label} ---", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        print(f"  ({time.time() - self.t0:.2f}s)", flush=True)


def main() -> int:
    con = connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")
    print("loading views.sql...", flush=True)
    t0 = time.time()
    load_views(con)
    print(f"  loaded in {time.time() - t0:.2f}s", flush=True)

    failures: list[str] = []

    with timed("row counts"):
        rt = con.sql("SELECT count(*) FROM raw_trades").fetchone()[0]
        mt = con.sql("SELECT count(*) FROM markets_tokens").fetchone()[0]
        jf = con.sql("SELECT count(*) FROM joined_fills").fetchone()[0]
        tao = con.sql("SELECT count(*) FROM trader_actions_orphan").fetchone()[0]
        print(f"  raw_trades              : {rt:>14,}")
        print(f"  markets_tokens (table)  : {mt:>14,}")
        print(f"  joined_fills            : {jf:>14,}  (after self-trade & orphan & token-drift exclusions)")
        print(f"  trader_actions estimate : {jf*2:>14,}  (= 2 × joined_fills)")
        print(f"  trader_actions_orphan   : {tao:>14,}")
        if jf < rt * 0.95 or jf > rt:
            failures.append(f"joined_fills row count out of expected band: {jf} vs raw {rt}")

    with timed("domah via trader_actions (single-address filter)"):
        domah_df = con.sql(f"""
            SELECT role, count(*) AS fills, round(sum(usd_amount), 2) AS usd_volume,
                   count(distinct market_id) AS distinct_markets,
                   min(timestamp) AS earliest, max(timestamp) AS latest
            FROM trader_actions WHERE address = '{DOMAH}'
            GROUP BY role ORDER BY role
        """).fetchdf()
        print(domah_df.to_string(index=False))
        domah_total = con.sql(
            f"SELECT count(*), count(distinct market_id) FROM trader_actions WHERE address = '{DOMAH}'"
        ).fetchone()
        print(f"\n  total fills    : {domah_total[0]:>10,}  (validation report: 645,772)")
        print(f"  distinct mkts  : {domah_total[1]:>10,}  (validation report: 9,444)")
        # trader_actions filters drop self-trades + orphans (~3k for domah)
        if abs(domah_total[0] - 645_772) > 645_772 * 0.01:
            failures.append(
                f"domah fill count {domah_total[0]} differs from validation by >1% (expected ~645,772)"
            )

    with timed("sign symmetry on 5 specific transactions"):
        # Pick 5 sample tx hashes via a tiny scoped scan (one shard, limit 5)
        sample_tx = [
            r[0] for r in con.sql("""
                SELECT DISTINCT transaction_hash
                FROM read_parquet('data/trades/trades_delta_shard1_2025-10-15_2025-11-13.parquet')
                WHERE market_id IS NOT NULL
                LIMIT 5
            """).fetchall()
        ]
        print(f"  sampled txs: {[h[:12] for h in sample_tx]}")

        max_abs_tok = 0.0
        max_abs_usd = 0.0
        n_groups = 0
        n_asymm_tok = 0
        n_asymm_usd = 0
        for tx in sample_tx:
            df = con.sql(f"""
                SELECT transaction_hash, outcome_token_id,
                       sum(token_delta) AS sum_tok,
                       sum(usd_delta)   AS sum_usd,
                       count(*) AS n_actions
                FROM trader_actions
                WHERE transaction_hash = '{tx}'
                GROUP BY transaction_hash, outcome_token_id
            """).fetchdf()
            n_groups += len(df)
            for _, r in df.iterrows():
                if abs(r["sum_tok"]) > 1e-6:
                    n_asymm_tok += 1
                    max_abs_tok = max(max_abs_tok, abs(r["sum_tok"]))
                if abs(r["sum_usd"]) > 1e-3:
                    n_asymm_usd += 1
                    max_abs_usd = max(max_abs_usd, abs(r["sum_usd"]))

        print(f"  groups: {n_groups}, asymm_token: {n_asymm_tok}, asymm_usd: {n_asymm_usd}")
        print(f"  max |Σtoken_delta|: {max_abs_tok:.6f}, max |Σusd_delta|: {max_abs_usd:.4f}")
        if n_asymm_tok > 0:
            failures.append(f"{n_asymm_tok} groups have non-zero token_delta sum (max {max_abs_tok:.6f})")
        if n_asymm_usd > 0:
            failures.append(f"{n_asymm_usd} groups have non-zero usd_delta sum (max {max_abs_usd:.4f})")

    if failures:
        print("\nSMOKE TESTS FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
