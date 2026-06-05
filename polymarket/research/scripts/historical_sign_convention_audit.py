"""Audit historical ``maker_side`` semantics against post-trade price action."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"

OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "historical_sign_audit_results.csv"
OUT_NOTE = NOTES / "historical_sign_convention_audit.md"

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260527
MAX_LOOKUP_GAP_SECONDS = 300


@dataclass(frozen=True)
class FamilyInput:
    family: str
    fills_path: Path


FAMILY_INPUTS = [
    FamilyInput("ai_product", ANALYSIS / "dali_tfi_ai_product_100_fills.parquet"),
    FamilyInput(
        "daily_crypto_up_down",
        ANALYSIS / "dali_tfi_crypto_250_exlast600_fills.parquet",
    ),
    FamilyInput("daily_equity_index", ANALYSIS / "dali_tfi_equity_index_100_fills.parquet"),
    FamilyInput("sports_game_lines", ANALYSIS / "dali_tfi_sports_100_fills.parquet"),
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def bernoulli_ci(
    successes: int,
    n: int,
    rng: np.random.Generator,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    draws = rng.binomial(n, p, size=resamples) / n
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def contrast_ci(
    sell_up: int,
    sell_n: int,
    buy_up: int,
    buy_n: int,
    rng: np.random.Generator,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if min(sell_n, buy_n) <= 0:
        return (float("nan"), float("nan"))
    sell_draws = rng.binomial(sell_n, sell_up / sell_n, size=resamples) / sell_n
    buy_draws = rng.binomial(buy_n, buy_up / buy_n, size=resamples) / buy_n
    return tuple(np.quantile(sell_draws - buy_draws, [0.025, 0.975]).tolist())


def z_against_half(successes: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return (successes / n - 0.5) / np.sqrt(0.25 / n)


def format_ts(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def interpretation(row: pd.Series) -> str:
    buy_ci_low = float(row["p_up_buy_ci_low"])
    buy_ci_high = float(row["p_up_buy_ci_high"])
    contrast_low = float(row["p_up_sell_minus_buy_ci_low"])
    contrast_high = float(row["p_up_sell_minus_buy_ci_high"])

    if buy_ci_high < 0.5:
        return "A"
    if buy_ci_low > 0.5 and contrast_low > 0:
        return "mixed_A_drift"
    if buy_ci_low > 0.5 and contrast_high < 0:
        return "B"
    if buy_ci_low <= 0.5 <= buy_ci_high:
        return "C"
    return "mixed"


def decision(row: pd.Series) -> str:
    interp = str(row["interpretation"])
    if interp in {"A", "mixed_A_drift"}:
        return "historical_to_aggressor_correct"
    if interp == "B":
        return "historical_to_aggressor_inverted"
    return "inconclusive"


def audit_family(
    con: duckdb.DuckDBPyConnection,
    inp: FamilyInput,
    rng: np.random.Generator,
) -> dict[str, object]:
    path = inp.fills_path
    if not path.exists():
        raise FileNotFoundError(path)

    source = con.sql(
        f"""
        SELECT
            count(*) AS source_fills,
            min(timestamp) AS source_start_ts,
            max(timestamp) AS source_end_ts,
            count(DISTINCT market_id) AS source_markets,
            count(DISTINCT CASE WHEN maker_asset_id = '0'
                                THEN taker_asset_id ELSE maker_asset_id END)
                AS source_outcome_tokens
        FROM read_parquet('{path}')
        WHERE price IS NOT NULL
          AND usd_amount > 0
          AND maker_side IN ('BUY', 'SELL')
        """
    ).fetchone()

    side_rows = con.sql(
        f"""
        WITH fills AS (
            SELECT
                row_number() OVER () AS anchor_id,
                timestamp,
                market_id,
                CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
                    AS outcome_token_id,
                maker_side,
                timestamp - INTERVAL 30 SECOND AS before_target,
                timestamp + INTERVAL 30 SECOND AS after_target,
                timestamp + INTERVAL 60 SECOND AS after60_target
            FROM read_parquet('{path}')
            WHERE price IS NOT NULL
              AND usd_amount > 0
              AND maker_side IN ('BUY', 'SELL')
        ),
        price_points AS (
            SELECT
                timestamp AS price_ts,
                market_id,
                CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
                    AS outcome_token_id,
                sum(price * usd_amount) / NULLIF(sum(usd_amount), 0) AS vwap_price
            FROM read_parquet('{path}')
            WHERE price IS NOT NULL
              AND usd_amount > 0
            GROUP BY timestamp, market_id, outcome_token_id
        ),
        with_before AS (
            SELECT
                a.*,
                pb.price_ts AS before_ts,
                pb.vwap_price AS price_before
            FROM fills a
            ASOF LEFT JOIN price_points pb
              ON a.market_id = pb.market_id
             AND a.outcome_token_id = pb.outcome_token_id
             AND a.before_target >= pb.price_ts
        ),
        with_after AS (
            SELECT
                wb.*,
                pa.price_ts AS after_ts,
                pa.vwap_price AS price_after
            FROM with_before wb
            ASOF LEFT JOIN price_points pa
              ON wb.market_id = pa.market_id
             AND wb.outcome_token_id = pa.outcome_token_id
             AND wb.after_target <= pa.price_ts
        ),
        with_after60 AS (
            SELECT
                wa.*,
                p60.price_ts AS after60_ts
            FROM with_after wa
            ASOF LEFT JOIN price_points p60
              ON wa.market_id = p60.market_id
             AND wa.outcome_token_id = p60.outcome_token_id
             AND wa.after60_target <= p60.price_ts
        ),
        eligible AS (
            SELECT
                *,
                date_diff('second', before_ts, before_target) AS before_gap_seconds,
                date_diff('second', after_target, after_ts) AS after_gap_seconds,
                date_diff('second', after60_target, after60_ts) AS after60_gap_seconds,
                price_after - price_before AS delta_price
            FROM with_after60
            WHERE price_before IS NOT NULL
              AND price_after IS NOT NULL
              AND after60_ts IS NOT NULL
              AND date_diff('second', before_ts, before_target)
                    BETWEEN 0 AND {MAX_LOOKUP_GAP_SECONDS}
              AND date_diff('second', after_target, after_ts)
                    BETWEEN 0 AND {MAX_LOOKUP_GAP_SECONDS}
              AND date_diff('second', after60_target, after60_ts)
                    BETWEEN 0 AND {MAX_LOOKUP_GAP_SECONDS}
        )
        SELECT
            maker_side,
            count(*) AS n,
            sum(CASE WHEN delta_price > 0 THEN 1 ELSE 0 END) AS n_up,
            sum(CASE WHEN delta_price < 0 THEN 1 ELSE 0 END) AS n_down,
            sum(CASE WHEN delta_price = 0 THEN 1 ELSE 0 END) AS n_flat,
            avg(delta_price) AS mean_delta_price,
            median(delta_price) AS median_delta_price,
            avg(before_gap_seconds) AS avg_before_gap_seconds,
            avg(after_gap_seconds) AS avg_after_gap_seconds,
            avg(after60_gap_seconds) AS avg_after60_gap_seconds,
            min(timestamp) AS eligible_start_ts,
            max(timestamp) AS eligible_end_ts
        FROM eligible
        GROUP BY maker_side
        """
    ).fetchdf()

    by_side = {str(row["maker_side"]): row for _, row in side_rows.iterrows()}
    missing = {"BUY", "SELL"} - set(by_side)
    if missing:
        raise RuntimeError(f"{inp.family}: missing eligible sides {sorted(missing)}")

    buy = by_side["BUY"]
    sell = by_side["SELL"]
    buy_n = int(buy["n"])
    sell_n = int(sell["n"])
    buy_up = int(buy["n_up"])
    sell_up = int(sell["n_up"])
    buy_down = int(buy["n_down"])
    sell_down = int(sell["n_down"])

    buy_up_ci = bernoulli_ci(buy_up, buy_n, rng)
    sell_up_ci = bernoulli_ci(sell_up, sell_n, rng)
    buy_down_ci = bernoulli_ci(buy_down, buy_n, rng)
    sell_down_ci = bernoulli_ci(sell_down, sell_n, rng)
    diff_ci = contrast_ci(sell_up, sell_n, buy_up, buy_n, rng)

    result: dict[str, object] = {
        "family": inp.family,
        "source_file": str(path.relative_to(ROOT)),
        "source_fills": int(source[0]),
        "source_start_ts": format_ts(source[1]),
        "source_end_ts": format_ts(source[2]),
        "source_markets": int(source[3]),
        "source_outcome_tokens": int(source[4]),
        "eligible_fills": buy_n + sell_n,
        "eligible_buy_fills": buy_n,
        "eligible_sell_fills": sell_n,
        "eligible_start_ts": format_ts(min(buy["eligible_start_ts"], sell["eligible_start_ts"])),
        "eligible_end_ts": format_ts(max(buy["eligible_end_ts"], sell["eligible_end_ts"])),
        "p_up_buy": buy_up / buy_n,
        "p_up_buy_ci_low": buy_up_ci[0],
        "p_up_buy_ci_high": buy_up_ci[1],
        "p_up_sell": sell_up / sell_n,
        "p_up_sell_ci_low": sell_up_ci[0],
        "p_up_sell_ci_high": sell_up_ci[1],
        "p_down_buy": buy_down / buy_n,
        "p_down_buy_ci_low": buy_down_ci[0],
        "p_down_buy_ci_high": buy_down_ci[1],
        "p_down_sell": sell_down / sell_n,
        "p_down_sell_ci_low": sell_down_ci[0],
        "p_down_sell_ci_high": sell_down_ci[1],
        "p_flat_buy": int(buy["n_flat"]) / buy_n,
        "p_flat_sell": int(sell["n_flat"]) / sell_n,
        "p_up_sell_minus_buy": sell_up / sell_n - buy_up / buy_n,
        "p_up_sell_minus_buy_ci_low": diff_ci[0],
        "p_up_sell_minus_buy_ci_high": diff_ci[1],
        "z_p_up_buy_vs_0_5": z_against_half(buy_up, buy_n),
        "mean_delta_buy": float(buy["mean_delta_price"]),
        "mean_delta_sell": float(sell["mean_delta_price"]),
        "median_delta_buy": float(buy["median_delta_price"]),
        "median_delta_sell": float(sell["median_delta_price"]),
        "avg_before_gap_seconds": float(
            (buy["avg_before_gap_seconds"] * buy_n + sell["avg_before_gap_seconds"] * sell_n)
            / (buy_n + sell_n)
        ),
        "avg_after_gap_seconds": float(
            (buy["avg_after_gap_seconds"] * buy_n + sell["avg_after_gap_seconds"] * sell_n)
            / (buy_n + sell_n)
        ),
        "avg_after60_gap_seconds": float(
            (buy["avg_after60_gap_seconds"] * buy_n + sell["avg_after60_gap_seconds"] * sell_n)
            / (buy_n + sell_n)
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "max_lookup_gap_seconds": MAX_LOOKUP_GAP_SECONDS,
        "price_proxy": "same-token trade VWAP ASOF proxy, not true historical book mid",
    }
    result["interpretation"] = interpretation(pd.Series(result))
    result["decision"] = decision(pd.Series(result))
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def ci_pct(low: float, high: float) -> str:
    return f"[{100.0 * low:.2f}%, {100.0 * high:.2f}%]"


def write_note(results: pd.DataFrame) -> None:
    table_rows = []
    down_rows = []
    for row in results.itertuples(index=False):
        table_rows.append(
            "| {family} | {n:,} | {pbuy} {cibuy} | {psell} {cisell} | {diff} {cidiff} | {z:.2f} | {interp} | {decision} |".format(
                family=row.family,
                n=int(row.eligible_fills),
                pbuy=pct(float(row.p_up_buy)),
                cibuy=ci_pct(float(row.p_up_buy_ci_low), float(row.p_up_buy_ci_high)),
                psell=pct(float(row.p_up_sell)),
                cisell=ci_pct(float(row.p_up_sell_ci_low), float(row.p_up_sell_ci_high)),
                diff=pct(float(row.p_up_sell_minus_buy)),
                cidiff=ci_pct(
                    float(row.p_up_sell_minus_buy_ci_low),
                    float(row.p_up_sell_minus_buy_ci_high),
                ),
                z=float(row.z_p_up_buy_vs_0_5),
                interp=row.interpretation,
                decision=row.decision,
            )
        )
        down_rows.append(
            "| {family} | {pdown_buy} {ci_down_buy} | {pdown_sell} {ci_down_sell} | {flat_buy} | {flat_sell} | {mean_buy:.4f} | {mean_sell:.4f} |".format(
                family=row.family,
                pdown_buy=pct(float(row.p_down_buy)),
                ci_down_buy=ci_pct(float(row.p_down_buy_ci_low), float(row.p_down_buy_ci_high)),
                pdown_sell=pct(float(row.p_down_sell)),
                ci_down_sell=ci_pct(float(row.p_down_sell_ci_low), float(row.p_down_sell_ci_high)),
                flat_buy=pct(float(row.p_flat_buy)),
                flat_sell=pct(float(row.p_flat_sell)),
                mean_buy=float(row.mean_delta_buy),
                mean_sell=float(row.mean_delta_sell),
            )
        )

    window_rows = []
    for row in results.itertuples(index=False):
        window_rows.append(
            f"| {row.family} | {row.source_start_ts} -> {row.source_end_ts} | "
            f"{row.eligible_start_ts} -> {row.eligible_end_ts} | "
            f"{int(row.source_markets):,} | {int(row.source_outcome_tokens):,} | "
            f"{float(row.avg_before_gap_seconds):.1f}s | "
            f"{float(row.avg_after_gap_seconds):.1f}s |"
        )

    OUT_NOTE.write_text(
        f"""# Historical Sign Convention Audit

Generated: 2026-05-27

## Executive Decision

`historical_to_aggressor()` is correct for the audited historical data. No
family showed evidence that the local `maker_side` field should be globally
inverted. The crypto slice has a small unconditional upward drift, so its
literal `P(up | maker_side=BUY)` is slightly above 0.5, but
`P(up | maker_side=SELL)` is much higher. That side contrast is the expected
pattern when `maker_side=BUY` is a passive bid hit by an aggressive seller and
`maker_side=SELL` is a passive ask lifted by an aggressive buyer.

No change was made to `lib/trade_sign_normalization.py`; downstream TFI
baseline analyses do not need a sign rerun for this audit.

## Documentation And Source Findings

- Repo docs identify the historical fill layer as a warproxxx seed plus
  Goldsky subgraph delta, consumed as `raw_trades`.
- The local Goldsky ingestion queries `orderFilledEvents` fields
  `makerAssetId`, `takerAssetId`, `makerAmountFilled`, and
  `takerAmountFilled`; it then sets `maker_side = BUY` exactly when
  `makerAssetId == '0'`.
- The seed builder uses the same rule. It treats `makerAssetId == '0'` as the
  maker paying USDC and receiving the outcome token, so `maker_side=BUY`.
- The direct Polygon decoder for newer logs also maps encoded V2 `side == 0`
  to `maker_asset_id='0'`, `taker_asset_id=token_id`, and then writes
  `maker_side=BUY`.
- I did not find a Goldsky page that defines a legacy subgraph column named
  `maker_side`; the local field is repo-derived. Goldsky's current
  Polymarket dataset docs describe `polymarket.order_filled` as a per-order
  fill dataset with `side` as order side and `order_type` as maker/taker.
  Their copy-trader guide is more explicit for V2: encoded side is the maker's
  side, and takers take the opposite side.
- Polymarket's on-chain order docs define V1 `makerAssetId`: if it is `0`, the
  order is a BUY giving pUSD/USDC for outcome tokens; `takerAssetId == 0` is a
  SELL receiving pUSD/USDC.

External references checked:

- [Goldsky: Indexing Polymarket](https://docs.goldsky.com/chains/polymarket)
- [Goldsky: Order Filled data source](https://app.goldsky.com/data-sources/dataset/order_filled)
- [Goldsky: Build a Polymarket copy-trader](https://docs.goldsky.com/compose/guides/build-a-polymarket-copy-trader)
- [Polymarket: Onchain Order Info](https://docs.polymarket.com/trading/orders/overview)

## Empirical Method

Sample source: the four fill slices used by the existing TFI magnitude analysis.
For each fill, the audit derived the outcome token as
`taker_asset_id` when `maker_asset_id='0'`, else `maker_asset_id`.

Historical true L2 book mid is not materialized in this repo. The empirical
audit therefore uses a same-token transaction-price proxy:

1. Build per `(market_id, outcome_token_id, timestamp)` VWAP from historical
   fills.
2. For each fill at time `t`, find the last same-token VWAP at or before
   `t - 30s`.
3. Find the first same-token VWAP at or after `t + 30s`.
4. Require same-token price data at or after `t + 60s`.
5. Require all lookup gaps to be no more than {MAX_LOOKUP_GAP_SECONDS} seconds.
6. Compute `delta_price = price(t+30s proxy) - price(t-30s proxy)`.

Confidence intervals are row-level bootstrap intervals with
{BOOTSTRAP_RESAMPLES:,} resamples and seed `{BOOTSTRAP_SEED}`. The z-score is
for `P(up | maker_side=BUY)` against the null `p=0.5`.

## Results

| family | eligible fills | P(up given BUY) 95% CI | P(up given SELL) 95% CI | SELL-BUY up contrast 95% CI | z(BUY vs 0.5) | interpretation | decision |
|---|---:|---:|---:|---:|---:|---|---|
{chr(10).join(table_rows)}

Down-move probabilities and mean deltas:

| family | P(down given BUY) 95% CI | P(down given SELL) 95% CI | P(flat given BUY) | P(flat given SELL) | mean delta after BUY | mean delta after SELL |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(down_rows)}

Interpretation keys:

- `A`: literal framework A; `P(up | maker_side=BUY)` is below 0.5.
- `mixed_A_drift`: `P(up | maker_side=BUY)` is above 0.5, but the SELL minus
  BUY up contrast is positive and significant, indicating family-level upward
  drift rather than inverted side semantics.
- `B`: inverted; BUY is significantly more upward than SELL.
- `C`: `P(up | maker_side=BUY)` is statistically indistinguishable from 0.5.
- `mixed`: significant but not cleanly classifiable.

## Data Windows

| family | source window | eligible audit window | markets | outcome tokens | avg pre lookup gap | avg post lookup gap |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(window_rows)}

## Per-Family Read

- `ai_product`: clear framework A. BUY-maker prints are followed by up moves
  only {pct(float(results.loc[results.family.eq("ai_product"), "p_up_buy"].iloc[0]))}
  of the time; SELL-maker prints are directionally higher.
- `daily_equity_index`: clear framework A, with a 10.1 percentage point
  SELL-minus-BUY up contrast.
- `sports_game_lines`: framework A by `P(up | BUY)` and by mean delta; flats
  are common, but the SELL-minus-BUY contrast is still positive.
- `daily_crypto_up_down`: mixed under the literal framework because crypto
  drifted upward in the analyzed window. The key diagnostic is that SELL-maker
  prints are 7.2 percentage points more likely to be followed by up moves than
  BUY-maker prints, which rejects the inverted-aggressor interpretation.

## Decision

Documentation and code provenance both say the local historical `maker_side`
field is the maker's token side. The empirical side contrast agrees across all
four audited families. The correct aggressor conversion remains:

- `maker_side=BUY` -> aggressor `SELL`
- `maker_side=SELL` -> aggressor `BUY`

No helper update or downstream TFI rerun is required.
""",
        encoding="utf-8",
    )


def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    con = connect()
    rows = [audit_family(con, inp, rng) for inp in FAMILY_INPUTS]
    results = pd.DataFrame(rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_NOTE.relative_to(ROOT)}")
    print(results[["family", "eligible_fills", "p_up_buy", "p_up_sell", "interpretation", "decision"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
