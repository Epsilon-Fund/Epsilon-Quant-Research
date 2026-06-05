"""K4 intra-Polymarket rebalancing and combinatorial arbitrage scan.

Research-only sidecar. It uses the canonical A1 feature parquet, which now
contains all owned captures (a0, a0b, a0c, a0c_roll), and writes interval-level
opportunities plus a compact findings note.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k4_combinatorial_arb.csv"
NOTE = NOTES / "block_k4_arb_scan_findings.md"

SNAPSHOT_EVENTS = ("book", "price_change", "best_bid_ask")
MAX_STALE_BOOK_SECONDS = 5.0
EPS = 1e-9
MIN_EXECUTABLE_SIZE = 5.0
MEDIAN_ACTION_LATENCY_MS = 300.0
P99_ACTION_LATENCY_MS = 900.0

OPPORTUNITY_COLUMNS = [
    "opportunity_type",
    "arb_direction",
    "execution_model",
    "run_id",
    "family",
    "market_id",
    "market_ids",
    "linked_group",
    "slug",
    "slugs",
    "question",
    "start_ts",
    "end_ts",
    "duration_ms",
    "duration_sec",
    "n_state_rows",
    "n_linked_markets",
    "max_gap_cents",
    "mean_gap_cents",
    "max_size_shares",
    "min_size_shares",
    "median_size_shares",
    "max_gross_profit_usd",
    "mean_gross_profit_usd",
    "max_capital_required_usd",
    "mean_pair_spread_cents",
    "spread_regime",
    "depth_regime",
    "time_to_resolution_regime",
    "time_to_resolution_sec",
    "clock_hour_utc",
    "capturable_median_latency",
    "capturable_p99_latency",
    "primary_capturable",
    "meets_min_size",
    "action_latency_ms_primary",
    "source",
]


@dataclass(frozen=True)
class LogicalGroup:
    run_id: str
    group_type: str
    group_name: str
    market_ids: tuple[str, ...]
    relation: str
    family: str
    slugs: tuple[str, ...]
    questions: tuple[str, ...]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_text(value: object, max_len: int = 82) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}"


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def bucket_spread(mean_pair_spread_cents: float) -> str:
    if pd.isna(mean_pair_spread_cents):
        return "unknown"
    if mean_pair_spread_cents <= 2.0:
        return "tight_<=2c"
    if mean_pair_spread_cents <= 5.0:
        return "normal_2_5c"
    return "wide_>5c"


def bucket_depth(size_shares: float) -> str:
    if pd.isna(size_shares):
        return "unknown"
    if size_shares < MIN_EXECUTABLE_SIZE:
        return "dust_<5"
    if size_shares < 25.0:
        return "small_5_25"
    if size_shares < 100.0:
        return "medium_25_100"
    return "deep_>=100"


def bucket_ttr(seconds: float) -> str:
    if pd.isna(seconds) or not np.isfinite(float(seconds)):
        return "unknown"
    if seconds < 0:
        return "resolved_or_past"
    if seconds <= 3600:
        return "<=1h"
    if seconds <= 6 * 3600:
        return "1_6h"
    if seconds <= 24 * 3600:
        return "6_24h"
    if seconds <= 7 * 24 * 3600:
        return "1_7d"
    return ">7d"


def stable_join(values: Iterable[object]) -> str:
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in out:
            out.append(text)
    return ";".join(out)


def prepare_market_states(con: duckdb.DuckDBPyConnection) -> None:
    events = ", ".join(f"'{event}'" for event in SNAPSHOT_EVENTS)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE k4_market_states AS
        WITH filtered AS (
            SELECT
                row_number() OVER () AS source_row,
                CAST(run_id AS VARCHAR) AS run_id,
                CAST(market_id AS VARCHAR) AS market_id,
                COALESCE(CAST(family AS VARCHAR), '') AS family,
                COALESCE(CAST(slug AS VARCHAR), '') AS slug,
                COALESCE(CAST(question AS VARCHAR), '') AS question,
                CAST(outcome_index AS INTEGER) AS outcome_index,
                exchange_ts AS event_ts,
                market_resolved_at AS market_resolved_at,
                CAST(best_bid AS DOUBLE) AS best_bid,
                CAST(best_bid_size AS DOUBLE) AS best_bid_size,
                CAST(best_ask AS DOUBLE) AS best_ask,
                CAST(best_ask_size AS DOUBLE) AS best_ask_size
            FROM read_parquet('{FEATURES}')
            WHERE event_type IN ({events})
              AND outcome_index IN (0, 1)
              AND exchange_ts IS NOT NULL
              AND COALESCE(is_book_state_complete, false)
              AND book_staleness_seconds <= {MAX_STALE_BOOK_SECONDS}
              AND best_bid IS NOT NULL
              AND best_ask IS NOT NULL
              AND best_bid_size IS NOT NULL
              AND best_ask_size IS NOT NULL
              AND best_bid >= 0.0
              AND best_bid <= 1.0
              AND best_ask >= 0.0
              AND best_ask <= 1.0
              AND best_ask >= best_bid
              AND best_bid_size > 0.0
              AND best_ask_size > 0.0
        ),
        market_meta AS (
            SELECT
                run_id,
                market_id,
                any_value(family) AS family,
                any_value(slug) AS slug,
                any_value(question) AS question,
                max(market_resolved_at) AS market_resolved_at
            FROM filtered
            GROUP BY run_id, market_id
        ),
        latest_by_ts AS (
            SELECT
                run_id,
                market_id,
                event_ts,
                outcome_index,
                arg_max(best_bid, source_row) AS best_bid,
                arg_max(best_bid_size, source_row) AS best_bid_size,
                arg_max(best_ask, source_row) AS best_ask,
                arg_max(best_ask_size, source_row) AS best_ask_size
            FROM filtered
            GROUP BY run_id, market_id, event_ts, outcome_index
        ),
        pivoted AS (
            SELECT
                run_id,
                market_id,
                event_ts,
                max(CASE WHEN outcome_index = 0 THEN best_bid END) AS yes_bid,
                max(CASE WHEN outcome_index = 0 THEN best_bid_size END) AS yes_bid_size,
                max(CASE WHEN outcome_index = 0 THEN best_ask END) AS yes_ask,
                max(CASE WHEN outcome_index = 0 THEN best_ask_size END) AS yes_ask_size,
                max(CASE WHEN outcome_index = 1 THEN best_bid END) AS no_bid,
                max(CASE WHEN outcome_index = 1 THEN best_bid_size END) AS no_bid_size,
                max(CASE WHEN outcome_index = 1 THEN best_ask END) AS no_ask,
                max(CASE WHEN outcome_index = 1 THEN best_ask_size END) AS no_ask_size
            FROM latest_by_ts
            GROUP BY run_id, market_id, event_ts
        ),
        state_raw AS (
            SELECT
                p.run_id,
                p.market_id,
                m.family,
                m.slug,
                m.question,
                m.market_resolved_at,
                p.event_ts,
                last_value(p.yes_bid IGNORE NULLS) OVER w AS yes_bid,
                last_value(p.yes_bid_size IGNORE NULLS) OVER w AS yes_bid_size,
                last_value(p.yes_ask IGNORE NULLS) OVER w AS yes_ask,
                last_value(p.yes_ask_size IGNORE NULLS) OVER w AS yes_ask_size,
                last_value(p.no_bid IGNORE NULLS) OVER w AS no_bid,
                last_value(p.no_bid_size IGNORE NULLS) OVER w AS no_bid_size,
                last_value(p.no_ask IGNORE NULLS) OVER w AS no_ask,
                last_value(p.no_ask_size IGNORE NULLS) OVER w AS no_ask_size
            FROM pivoted p
            JOIN market_meta m USING (run_id, market_id)
            WINDOW w AS (
                PARTITION BY p.run_id, p.market_id
                ORDER BY p.event_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            *,
            lead(event_ts) OVER (
                PARTITION BY run_id, market_id ORDER BY event_ts
            ) AS next_ts,
            (yes_ask - yes_bid) + (no_ask - no_bid) AS pair_spread,
            least(yes_ask_size, no_ask_size) AS ask_bundle_size,
            least(yes_bid_size, no_bid_size) AS bid_bundle_size
        FROM state_raw
        WHERE yes_bid IS NOT NULL
          AND yes_bid_size IS NOT NULL
          AND yes_ask IS NOT NULL
          AND yes_ask_size IS NOT NULL
          AND no_bid IS NOT NULL
          AND no_bid_size IS NOT NULL
          AND no_ask IS NOT NULL
          AND no_ask_size IS NOT NULL
        """
    )


def load_rebalancing_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        f"""
        WITH opps AS (
            SELECT
                'rebalancing' AS opportunity_type,
                'buy_yes_no_at_asks' AS arb_direction,
                'long_bundle_executable' AS execution_model,
                run_id,
                family,
                market_id,
                market_id AS market_ids,
                'binary_yes_no_parity' AS linked_group,
                slug,
                slug AS slugs,
                question,
                event_ts,
                next_ts,
                2 AS n_linked_markets,
                1.0 - (yes_ask + no_ask) AS profit_per_share,
                (1.0 - (yes_ask + no_ask)) * 100.0 AS gap_cents,
                ask_bundle_size AS size_available_shares,
                (1.0 - (yes_ask + no_ask)) * ask_bundle_size AS gross_profit_usd,
                (yes_ask + no_ask) * ask_bundle_size AS capital_required_usd,
                pair_spread * 100.0 AS pair_spread_cents,
                CASE
                    WHEN market_resolved_at IS NULL THEN NULL
                    ELSE epoch(market_resolved_at - event_ts)
                END AS time_to_resolution_sec,
                'top_of_book_replay' AS source
            FROM k4_market_states
            WHERE yes_ask + no_ask < 1.0 - {EPS}

            UNION ALL

            SELECT
                'rebalancing' AS opportunity_type,
                'mint_sell_yes_no_to_bids' AS arb_direction,
                'mint_then_sell_executable' AS execution_model,
                run_id,
                family,
                market_id,
                market_id AS market_ids,
                'binary_yes_no_parity' AS linked_group,
                slug,
                slug AS slugs,
                question,
                event_ts,
                next_ts,
                2 AS n_linked_markets,
                (yes_bid + no_bid) - 1.0 AS profit_per_share,
                ((yes_bid + no_bid) - 1.0) * 100.0 AS gap_cents,
                bid_bundle_size AS size_available_shares,
                ((yes_bid + no_bid) - 1.0) * bid_bundle_size AS gross_profit_usd,
                bid_bundle_size AS capital_required_usd,
                pair_spread * 100.0 AS pair_spread_cents,
                CASE
                    WHEN market_resolved_at IS NULL THEN NULL
                    ELSE epoch(market_resolved_at - event_ts)
                END AS time_to_resolution_sec,
                'top_of_book_replay' AS source
            FROM k4_market_states
            WHERE yes_bid + no_bid > 1.0 + {EPS}
        )
        SELECT * FROM opps
        ORDER BY run_id, market_id, arb_direction, event_ts
        """
    ).df()
    return normalize_event_times(df)


def normalize_event_times(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("event_ts", "next_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def collapse_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)
    rows = normalize_event_times(rows.copy())
    rows["next_ts"] = rows["next_ts"].fillna(rows["event_ts"])
    rows = rows.sort_values(
        ["opportunity_type", "run_id", "linked_group", "market_ids", "arb_direction", "event_ts"]
    ).reset_index(drop=True)

    collapsed: list[dict[str, object]] = []
    group_cols = [
        "opportunity_type",
        "arb_direction",
        "execution_model",
        "run_id",
        "family",
        "market_id",
        "market_ids",
        "linked_group",
        "slug",
        "slugs",
        "question",
        "n_linked_markets",
        "source",
    ]
    for _, group in rows.groupby(group_cols, dropna=False, sort=False):
        current: list[pd.Series] = []
        prev_next: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            series = pd.Series(row._asdict())
            event_ts = series["event_ts"]
            next_ts = series["next_ts"] if pd.notna(series["next_ts"]) else event_ts
            starts_new = bool(current) and (prev_next is pd.NaT or event_ts != prev_next)
            if starts_new:
                collapsed.append(collapse_interval(current))
                current = []
            current.append(series)
            prev_next = next_ts
        if current:
            collapsed.append(collapse_interval(current))

    out = pd.DataFrame(collapsed)
    if out.empty:
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)
    out["duration_sec"] = out["duration_ms"] / 1000.0
    out["meets_min_size"] = out["max_size_shares"].ge(MIN_EXECUTABLE_SIZE)
    out["capturable_median_latency"] = (
        out["duration_ms"].ge(MEDIAN_ACTION_LATENCY_MS) & out["meets_min_size"]
    )
    out["capturable_p99_latency"] = out["duration_ms"].ge(P99_ACTION_LATENCY_MS) & out["meets_min_size"]
    out["primary_capturable"] = out["capturable_p99_latency"]
    out["action_latency_ms_primary"] = P99_ACTION_LATENCY_MS
    out["spread_regime"] = out["mean_pair_spread_cents"].map(bucket_spread)
    out["depth_regime"] = out["max_size_shares"].map(bucket_depth)
    out["time_to_resolution_regime"] = out["time_to_resolution_sec"].map(bucket_ttr)
    out["clock_hour_utc"] = pd.to_datetime(out["start_ts"], utc=True).dt.hour
    for col in OPPORTUNITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[OPPORTUNITY_COLUMNS].sort_values(
        ["start_ts", "opportunity_type", "run_id", "linked_group", "arb_direction"]
    )


def collapse_interval(rows: list[pd.Series]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    start_ts = frame["event_ts"].iloc[0]
    end_ts = frame["next_ts"].iloc[-1]
    if pd.isna(end_ts):
        end_ts = frame["event_ts"].iloc[-1]
    duration_ms = max(0.0, (end_ts - start_ts).total_seconds() * 1000.0)
    state_durations = (frame["next_ts"] - frame["event_ts"]).dt.total_seconds().mul(1000.0)
    state_durations = state_durations.replace([np.inf, -np.inf], np.nan).clip(lower=0.0).fillna(0.0)
    weights = state_durations.to_numpy(dtype=float)

    def weighted_mean(col: str) -> float:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(values)
        if not ok.any():
            return math.nan
        if np.nansum(weights[ok]) > 0:
            return float(np.average(values[ok], weights=weights[ok]))
        return float(np.nanmean(values[ok]))

    ttr = weighted_mean("time_to_resolution_sec")
    if not np.isfinite(ttr):
        ttr = pd.to_numeric(frame["time_to_resolution_sec"], errors="coerce").dropna()
        ttr = float(ttr.iloc[0]) if not ttr.empty else math.nan

    out = {
        "opportunity_type": frame["opportunity_type"].iloc[0],
        "arb_direction": frame["arb_direction"].iloc[0],
        "execution_model": frame["execution_model"].iloc[0],
        "run_id": frame["run_id"].iloc[0],
        "family": frame["family"].iloc[0],
        "market_id": frame["market_id"].iloc[0],
        "market_ids": frame["market_ids"].iloc[0],
        "linked_group": frame["linked_group"].iloc[0],
        "slug": frame["slug"].iloc[0],
        "slugs": frame["slugs"].iloc[0],
        "question": frame["question"].iloc[0],
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "duration_ms": duration_ms,
        "n_state_rows": int(len(frame)),
        "n_linked_markets": int(frame["n_linked_markets"].iloc[0]),
        "max_gap_cents": float(pd.to_numeric(frame["gap_cents"], errors="coerce").max()),
        "mean_gap_cents": weighted_mean("gap_cents"),
        "max_size_shares": float(pd.to_numeric(frame["size_available_shares"], errors="coerce").max()),
        "min_size_shares": float(pd.to_numeric(frame["size_available_shares"], errors="coerce").min()),
        "median_size_shares": float(pd.to_numeric(frame["size_available_shares"], errors="coerce").median()),
        "max_gross_profit_usd": float(pd.to_numeric(frame["gross_profit_usd"], errors="coerce").max()),
        "mean_gross_profit_usd": weighted_mean("gross_profit_usd"),
        "max_capital_required_usd": float(
            pd.to_numeric(frame["capital_required_usd"], errors="coerce").max()
        ),
        "mean_pair_spread_cents": weighted_mean("pair_spread_cents"),
        "time_to_resolution_sec": ttr,
        "source": frame["source"].iloc[0],
    }
    return out


def load_market_meta(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            run_id,
            market_id,
            any_value(family) AS family,
            any_value(slug) AS slug,
            any_value(question) AS question,
            count(*) AS n_state_rows
        FROM k4_market_states
        GROUP BY run_id, market_id
        ORDER BY run_id, family, slug
        """
    ).df()


def build_logical_groups(meta: pd.DataFrame) -> list[LogicalGroup]:
    groups: list[LogicalGroup] = []
    for run_id, run_meta in meta.groupby("run_id", sort=False):
        groups.extend(exclusive_group(run_id, run_meta, "ai_best_model_june_2026", "best-ai-model-at-the-end-of-june"))
        groups.extend(exclusive_group(run_id, run_meta, "fifa_world_cup_2026_winner", "win the 2026 fifa world cup"))
        groups.extend(exclusive_group(run_id, run_meta, "champions_league_2025_26_winner", "2025", "champions league"))
        groups.extend(exclusive_group(run_id, run_meta, "nba_finals_2026_winner", "win the 2026 nba finals"))

        june = run_meta[
            run_meta["slug"].str.contains("strait-of-hormuz-traffic-returns-to-normal-by-end-of-june", case=False, na=False)
        ]
        july = run_meta[
            run_meta["slug"].str.contains("strait-of-hormuz-traffic-returns-to-normal-by-july-31", case=False, na=False)
        ]
        if len(june) == 1 and len(july) == 1:
            members = pd.concat([june, july], ignore_index=True)
            groups.append(
                LogicalGroup(
                    run_id=str(run_id),
                    group_type="implication",
                    group_name="strait_hormuz_june_implies_july",
                    market_ids=tuple(members["market_id"].astype(str)),
                    relation="market_0_yes_implies_market_1_yes",
                    family=stable_family(members["family"]),
                    slugs=tuple(members["slug"].astype(str)),
                    questions=tuple(members["question"].astype(str)),
                )
            )
    return groups


def exclusive_group(run_id: str, meta: pd.DataFrame, name: str, *needles: str) -> list[LogicalGroup]:
    hay = (meta["slug"].fillna("") + " " + meta["question"].fillna("")).str.lower()
    mask = pd.Series(True, index=meta.index)
    for needle in needles:
        mask &= hay.str.contains(re.escape(needle.lower()), na=False)
    members = meta.loc[mask].drop_duplicates("market_id").copy()
    if len(members) < 2:
        return []
    return [
        LogicalGroup(
            run_id=str(run_id),
            group_type="exclusive",
            group_name=name,
            market_ids=tuple(members["market_id"].astype(str)),
            relation="mutually_exclusive_non_exhaustive",
            family=stable_family(members["family"]),
            slugs=tuple(members["slug"].astype(str)),
            questions=tuple(members["question"].astype(str)),
        )
    ]


def stable_family(families: Iterable[object]) -> str:
    vals = [str(fam) for fam in families if str(fam)]
    uniq = sorted(set(vals))
    return uniq[0] if len(uniq) == 1 else "mixed"


def load_candidate_states(con: duckdb.DuckDBPyConnection, groups: list[LogicalGroup]) -> pd.DataFrame:
    pairs = sorted({(group.run_id, market_id) for group in groups for market_id in group.market_ids})
    if not pairs:
        return pd.DataFrame()
    values = ", ".join(f"('{run_id}', '{market_id}')" for run_id, market_id in pairs)
    df = con.execute(
        f"""
        WITH candidates(run_id, market_id) AS (VALUES {values})
        SELECT s.*
        FROM k4_market_states s
        JOIN candidates c USING (run_id, market_id)
        ORDER BY s.run_id, s.market_id, s.event_ts
        """
    ).df()
    return normalize_event_times(df)


def build_group_state(states: pd.DataFrame, group: LogicalGroup) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for i, market_id in enumerate(group.market_ids):
        member = states[(states["run_id"].eq(group.run_id)) & (states["market_id"].eq(market_id))].copy()
        if member.empty:
            return pd.DataFrame()
        member = member.sort_values("event_ts").drop_duplicates("event_ts", keep="last")
        keep = [
            "event_ts",
            "yes_bid",
            "yes_bid_size",
            "yes_ask",
            "yes_ask_size",
            "no_bid",
            "no_bid_size",
            "no_ask",
            "no_ask_size",
            "pair_spread",
            "market_resolved_at",
        ]
        member = member[keep].set_index("event_ts")
        member = member.add_prefix(f"m{i}_")
        pieces.append(member)
    combined = pd.concat(pieces, axis=1).sort_index().ffill()
    required = []
    for i in range(len(group.market_ids)):
        required.extend(
            [
                f"m{i}_yes_bid",
                f"m{i}_yes_bid_size",
                f"m{i}_yes_ask",
                f"m{i}_yes_ask_size",
                f"m{i}_no_bid",
                f"m{i}_no_bid_size",
                f"m{i}_no_ask",
                f"m{i}_no_ask_size",
            ]
        )
    combined = combined.dropna(subset=required)
    if combined.empty:
        return combined
    combined["event_ts"] = combined.index
    combined["next_ts"] = combined["event_ts"].shift(-1)
    return combined.reset_index(drop=True)


def combinatorial_rows_for_group(states: pd.DataFrame, group: LogicalGroup) -> pd.DataFrame:
    frame = build_group_state(states, group)
    if frame.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    k = len(group.market_ids)

    if group.group_type == "exclusive":
        no_ask_sum = sum(frame[f"m{i}_no_ask"] for i in range(k))
        no_ask_size = np.minimum.reduce([frame[f"m{i}_no_ask_size"].to_numpy(dtype=float) for i in range(k)])
        min_payoff = float(k - 1)
        long_no_profit = min_payoff - no_ask_sum
        rows.append(
            make_group_rows(
                frame=frame,
                group=group,
                mask=long_no_profit > EPS,
                direction="exclusive_buy_all_no_asks",
                execution_model="long_bundle_executable",
                profit_per_share=long_no_profit,
                size_available_shares=no_ask_size,
                capital_required_usd=no_ask_sum * no_ask_size,
            )
        )

        yes_bid_sum = sum(frame[f"m{i}_yes_bid"] for i in range(k))
        yes_bid_size = np.minimum.reduce([frame[f"m{i}_yes_bid_size"].to_numpy(dtype=float) for i in range(k)])
        short_yes_profit = yes_bid_sum - 1.0
        rows.append(
            make_group_rows(
                frame=frame,
                group=group,
                mask=short_yes_profit > EPS,
                direction="exclusive_sell_all_yes_bids",
                execution_model="requires_short_or_inventory",
                profit_per_share=short_yes_profit,
                size_available_shares=yes_bid_size,
                capital_required_usd=yes_bid_size,
            )
        )

    elif group.group_type == "implication" and k == 2:
        antecedent = 0
        consequent = 1
        long_bundle_cost = frame[f"m{consequent}_yes_ask"] + frame[f"m{antecedent}_no_ask"]
        long_bundle_size = np.minimum(
            frame[f"m{consequent}_yes_ask_size"].to_numpy(dtype=float),
            frame[f"m{antecedent}_no_ask_size"].to_numpy(dtype=float),
        )
        long_profit = 1.0 - long_bundle_cost
        rows.append(
            make_group_rows(
                frame=frame,
                group=group,
                mask=long_profit > EPS,
                direction="implication_buy_consequent_yes_and_antecedent_no",
                execution_model="long_bundle_executable",
                profit_per_share=long_profit,
                size_available_shares=long_bundle_size,
                capital_required_usd=long_bundle_cost * long_bundle_size,
            )
        )

        yes_cross_profit = frame[f"m{antecedent}_yes_bid"] - frame[f"m{consequent}_yes_ask"]
        yes_cross_size = np.minimum(
            frame[f"m{antecedent}_yes_bid_size"].to_numpy(dtype=float),
            frame[f"m{consequent}_yes_ask_size"].to_numpy(dtype=float),
        )
        rows.append(
            make_group_rows(
                frame=frame,
                group=group,
                mask=yes_cross_profit > EPS,
                direction="implication_sell_antecedent_yes_buy_consequent_yes",
                execution_model="requires_short_or_inventory",
                profit_per_share=yes_cross_profit,
                size_available_shares=yes_cross_size,
                capital_required_usd=frame[f"m{consequent}_yes_ask"] * yes_cross_size,
            )
        )

        no_cross_profit = frame[f"m{consequent}_no_bid"] - frame[f"m{antecedent}_no_ask"]
        no_cross_size = np.minimum(
            frame[f"m{consequent}_no_bid_size"].to_numpy(dtype=float),
            frame[f"m{antecedent}_no_ask_size"].to_numpy(dtype=float),
        )
        rows.append(
            make_group_rows(
                frame=frame,
                group=group,
                mask=no_cross_profit > EPS,
                direction="implication_sell_consequent_no_buy_antecedent_no",
                execution_model="requires_short_or_inventory",
                profit_per_share=no_cross_profit,
                size_available_shares=no_cross_size,
                capital_required_usd=frame[f"m{antecedent}_no_ask"] * no_cross_size,
            )
        )

    valid = [row for row in rows if not row.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def make_group_rows(
    frame: pd.DataFrame,
    group: LogicalGroup,
    mask: pd.Series | np.ndarray,
    direction: str,
    execution_model: str,
    profit_per_share: pd.Series,
    size_available_shares: pd.Series | np.ndarray,
    capital_required_usd: pd.Series | np.ndarray,
) -> pd.DataFrame:
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    if not mask.any():
        return pd.DataFrame()
    out = frame.loc[mask, ["event_ts", "next_ts"]].copy()
    size = pd.Series(size_available_shares, index=frame.index).loc[mask].astype(float)
    profit = pd.Series(profit_per_share, index=frame.index).loc[mask].astype(float)
    capital = pd.Series(capital_required_usd, index=frame.index).loc[mask].astype(float)
    pair_spread = sum(frame[f"m{i}_pair_spread"] for i in range(len(group.market_ids))).loc[mask]

    resolved_times = []
    for i in range(len(group.market_ids)):
        col = pd.to_datetime(frame.loc[mask, f"m{i}_market_resolved_at"], utc=True, errors="coerce")
        resolved_times.append(col)
    resolved = pd.concat(resolved_times, axis=1).min(axis=1)
    ttr = (resolved - out["event_ts"]).dt.total_seconds()

    out["opportunity_type"] = "combinatorial"
    out["arb_direction"] = direction
    out["execution_model"] = execution_model
    out["run_id"] = group.run_id
    out["family"] = group.family
    out["market_id"] = ""
    out["market_ids"] = ";".join(group.market_ids)
    out["linked_group"] = group.group_name
    out["slug"] = ""
    out["slugs"] = ";".join(group.slugs)
    out["question"] = " / ".join(compact_text(question, 80) for question in group.questions)
    out["n_linked_markets"] = len(group.market_ids)
    out["profit_per_share"] = profit
    out["gap_cents"] = profit * 100.0
    out["size_available_shares"] = size
    out["gross_profit_usd"] = profit * size
    out["capital_required_usd"] = capital
    out["pair_spread_cents"] = pair_spread * 100.0
    out["time_to_resolution_sec"] = ttr
    out["source"] = f"heuristic_{group.relation}"
    return out


def load_row_count_heatmap(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            run_id,
            family,
            count(*) AS state_rows,
            count(DISTINCT market_id) AS markets,
            min(event_ts) AS first_event_ts,
            max(event_ts) AS last_event_ts
        FROM k4_market_states
        GROUP BY run_id, family
        ORDER BY run_id, family
        """
    ).df()


def load_run_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH raw AS (
            SELECT
                run_id,
                count(*) AS raw_rows,
                count(DISTINCT market_id) AS raw_markets,
                min(exchange_ts) AS raw_first_ts,
                max(exchange_ts) AS raw_last_ts
            FROM read_parquet('{FEATURES}')
            GROUP BY run_id
        ),
        states AS (
            SELECT
                run_id,
                count(*) AS state_rows,
                count(DISTINCT market_id) AS state_markets,
                min(event_ts) AS state_first_ts,
                max(event_ts) AS state_last_ts
            FROM k4_market_states
            GROUP BY run_id
        )
        SELECT *
        FROM raw
        LEFT JOIN states USING (run_id)
        ORDER BY run_id
        """
    ).df()


def summarize_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(cols, dropna=False)
        .agg(
            intervals=("opportunity_type", "size"),
            capturable_p99=("primary_capturable", "sum"),
            capturable_median=("capturable_median_latency", "sum"),
            max_gap_cents=("max_gap_cents", "max"),
            max_size_shares=("max_size_shares", "max"),
            max_gross_profit_usd=("max_gross_profit_usd", "max"),
            total_interval_duration_sec=("duration_sec", "sum"),
        )
        .reset_index()
        .sort_values(["capturable_p99", "intervals"], ascending=False)
    )
    return out


def write_note(
    opportunities: pd.DataFrame,
    groups: list[LogicalGroup],
    row_heatmap: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    primary = opportunities[opportunities["primary_capturable"].fillna(False)].copy()
    executable = opportunities[opportunities["execution_model"].str.contains("executable", na=False)].copy()
    primary_executable = primary[primary["execution_model"].str.contains("executable", na=False)].copy()
    rebalancing_exec = primary_executable[primary_executable["opportunity_type"].eq("rebalancing")]
    combinatorial_exec = primary_executable[primary_executable["opportunity_type"].eq("combinatorial")]

    if rebalancing_exec.empty and combinatorial_exec.empty:
        headline = (
            "No. On this owned universe, there were no p99-latency-capturable, "
            "minimum-size executable arb intervals after complete-book/fresh-book filtering."
        )
    else:
        headline = (
            "Maybe, but only as a scoped latency-arb thread: the scan found "
            f"{len(primary_executable):,} p99-latency-capturable executable intervals "
            f"({len(rebalancing_exec):,} rebalancing, {len(combinatorial_exec):,} combinatorial)."
        )

    summary_rows = []
    for _, row in summarize_by(opportunities, ["opportunity_type", "arb_direction", "execution_model"]).iterrows():
        summary_rows.append(
            [
                row["opportunity_type"],
                row["arb_direction"],
                row["execution_model"],
                fmt_int(row["intervals"]),
                fmt_int(row["capturable_p99"]),
                fmt_float(row["max_gap_cents"], 3),
                fmt_float(row["max_size_shares"], 2),
                fmt_float(row["max_gross_profit_usd"], 4),
            ]
        )

    family_rows = []
    for _, row in summarize_by(opportunities, ["family"]).head(20).iterrows():
        family_rows.append(
            [
                row["family"],
                fmt_int(row["intervals"]),
                fmt_int(row["capturable_p99"]),
                fmt_float(row["max_gap_cents"], 3),
                fmt_float(row["max_size_shares"], 2),
                fmt_float(row["total_interval_duration_sec"], 1),
            ]
        )

    segment_rows = []
    for _, row in summarize_by(opportunities, ["spread_regime", "depth_regime", "time_to_resolution_regime"]).head(20).iterrows():
        segment_rows.append(
            [
                row["spread_regime"],
                row["depth_regime"],
                row["time_to_resolution_regime"],
                fmt_int(row["intervals"]),
                fmt_int(row["capturable_p99"]),
                fmt_float(row["max_gap_cents"], 3),
            ]
        )

    clock_rows = []
    for _, row in summarize_by(opportunities, ["clock_hour_utc"]).head(24).sort_values("clock_hour_utc").iterrows():
        clock_rows.append(
            [
                fmt_int(row["clock_hour_utc"]),
                fmt_int(row["intervals"]),
                fmt_int(row["capturable_p99"]),
                fmt_float(row["max_gap_cents"], 3),
            ]
        )

    market_balance = []
    if not opportunities.empty:
        mb = summarize_by(opportunities, ["run_id", "market_ids", "linked_group", "slugs"]).head(12)
        for _, row in mb.iterrows():
            market_balance.append(
                [
                    row["run_id"],
                    compact_text(row["linked_group"], 34),
                    compact_text(row["slugs"], 58),
                    fmt_int(row["intervals"]),
                    fmt_int(row["capturable_p99"]),
                    fmt_float(row["max_gap_cents"], 3),
                    fmt_float(row["max_size_shares"], 2),
                ]
            )

    top_rows = []
    if not opportunities.empty:
        top = opportunities.sort_values(
            ["primary_capturable", "max_gross_profit_usd", "duration_ms"], ascending=False
        ).head(12)
        for _, row in top.iterrows():
            top_rows.append(
                [
                    row["opportunity_type"],
                    row["arb_direction"],
                    row["run_id"],
                    compact_text(row["slugs"] or row["slug"], 55),
                    fmt_float(row["duration_ms"], 0),
                    fmt_float(row["max_gap_cents"], 3),
                    fmt_float(row["max_size_shares"], 2),
                    str(bool(row["primary_capturable"])),
                ]
            )

    heatmap_rows = []
    for _, row in row_heatmap.iterrows():
        heatmap_rows.append(
            [
                row["run_id"],
                row["family"],
                fmt_int(row["markets"]),
                fmt_int(row["state_rows"]),
                str(pd.to_datetime(row["first_event_ts"], utc=True)),
                str(pd.to_datetime(row["last_event_ts"], utc=True)),
            ]
        )

    run_rows = []
    for _, row in run_summary.iterrows():
        run_rows.append(
            [
                row["run_id"],
                fmt_int(row["raw_rows"]),
                fmt_int(row["raw_markets"]),
                fmt_int(row["state_rows"]),
                fmt_int(row["state_markets"]),
                str(pd.to_datetime(row["state_first_ts"], utc=True)),
                str(pd.to_datetime(row["state_last_ts"], utc=True)),
            ]
        )

    group_rows = [
        [
            group.run_id,
            group.group_type,
            group.group_name,
            group.relation,
            "; ".join(compact_text(slug, 42) for slug in group.slugs),
        ]
        for group in groups
    ]

    total_intervals = len(opportunities)
    total_primary = int(opportunities["primary_capturable"].sum()) if not opportunities.empty else 0
    total_exec_primary = len(primary_executable)
    max_gap = opportunities["max_gap_cents"].max() if not opportunities.empty else math.nan
    max_size = opportunities["max_size_shares"].max() if not opportunities.empty else math.nan

    note = f"""# Block K4 — Intra-Polymarket Arb Scan

## Headline

**Is this a real parallel thread? {headline}**

Frequency × size: `{total_intervals:,}` violation intervals were detected in total, `{total_primary:,}` survived the primary p99 latency cut (`{P99_ACTION_LATENCY_MS:.0f}ms`) with top-of-book size at least `{MIN_EXECUTABLE_SIZE:g}` shares, and `{total_exec_primary:,}` intervals were both p99-capturable and executable long/mint bundles. The largest observed gap was `{fmt_float(max_gap, 3)}` cents per bundle unit; largest displayed size was `{fmt_float(max_size, 2)}` shares.

## Method

- Source: `{display_path(FEATURES)}`; this refreshed A1 feature parquet contains `a0`, `a0b`, `a0c`, and `a0c_roll`.
- Book controls: snapshot events only (`book`, `price_change`, `best_bid_ask`), exchange timestamps, complete book state, `book_staleness_seconds <= {MAX_STALE_BOOK_SECONDS:g}`, valid positive top-of-book sizes, and no crossed same-outcome books.
- Rebalancing rule: binary YES/NO ask sum `< $1` or bid sum `> $1`, collapsed into contiguous event-time intervals.
- Combinatorial rule: conservative owned-universe logical sets only. Mutually exclusive sets scan both buy-all-NO asks and sell-all-YES bids; the Strait of Hormuz date pair scans the June-implies-July implication.
- Capturable rule: primary headline uses `{P99_ACTION_LATENCY_MS:.0f}ms` action latency from `configs/backtest_default.yaml` p99 and top-of-book bundle size `>= {MIN_EXECUTABLE_SIZE:g}` shares. A `{MEDIAN_ACTION_LATENCY_MS:.0f}ms` median-latency flag is also in the CSV.
- Literature anchor: Saguillo et al., _Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets_ ([arXiv:2508.03474](https://arxiv.org/abs/2508.03474)), distinguishing market rebalancing and combinatorial arb.

## Opportunity Summary

{markdown_table(["type", "direction", "execution", "intervals", "p99 capturable", "max gap c", "max size", "max gross $"], summary_rows)}

## By Family

{markdown_table(["family", "intervals", "p99 capturable", "max gap c", "max size", "interval sec"], family_rows)}

## Segment Sensitivity

{markdown_table(["spread", "depth", "time to resolution", "intervals", "p99 capturable", "max gap c"], segment_rows)}

## Clock-Time Sensitivity

{markdown_table(["UTC hour", "intervals", "p99 capturable", "max gap c"], clock_rows)}

## Market-Balanced View

{markdown_table(["run", "group", "slugs", "intervals", "p99 capturable", "max gap c", "max size"], market_balance)}

## Largest Intervals

{markdown_table(["type", "direction", "run", "slugs", "duration ms", "max gap c", "max size", "p99"], top_rows)}

## Row-Count Heatmap

{markdown_table(["run", "family", "markets", "state rows", "first state", "last state"], heatmap_rows)}

## Run Coverage

{markdown_table(["run", "raw rows", "raw markets", "state rows", "state markets", "first state", "last state"], run_rows)}

## Logical Groups Considered

{markdown_table(["run", "type", "group", "relation", "slugs"], group_rows)}

## Read

This scan is deterministic and has no IS/OOS split. The key guardrail is that interval duration is measured from exchange-time book states, so single sparse ticks do not become fake capturable trades. Rebalancing rows are directly executable as complete-set buy/redeem or mint/sell bundles; combinatorial rows tagged `requires_short_or_inventory` are diagnostic unless we already hold the short leg inventory or can source it without leg risk.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(f"missing input parquet: {display_path(FEATURES)}")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    prepare_market_states(con)
    run_summary = load_run_summary(con)
    row_heatmap = load_row_count_heatmap(con)

    rebalancing_rows = load_rebalancing_rows(con)
    rebalancing = collapse_rows(rebalancing_rows)

    meta = load_market_meta(con)
    groups = build_logical_groups(meta)
    candidate_states = load_candidate_states(con, groups)
    combinatorial_parts = [combinatorial_rows_for_group(candidate_states, group) for group in groups]
    combinatorial_rows = (
        pd.concat([part for part in combinatorial_parts if not part.empty], ignore_index=True)
        if any(not part.empty for part in combinatorial_parts)
        else pd.DataFrame()
    )
    combinatorial = collapse_rows(combinatorial_rows)

    non_empty = [part for part in (rebalancing, combinatorial) if not part.empty]
    opportunities = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if opportunities.empty:
        opportunities = pd.DataFrame(columns=OPPORTUNITY_COLUMNS)
    else:
        opportunities = opportunities[OPPORTUNITY_COLUMNS].sort_values(
            ["start_ts", "opportunity_type", "run_id", "linked_group", "arb_direction"]
        )

    opportunities.to_csv(OUT_CSV, index=False)
    write_note(opportunities, groups, row_heatmap, run_summary)
    con.close()

    print(f"wrote {display_path(OUT_CSV)} rows={len(opportunities):,}")
    print(f"wrote {display_path(NOTE)}")


if __name__ == "__main__":
    main()
