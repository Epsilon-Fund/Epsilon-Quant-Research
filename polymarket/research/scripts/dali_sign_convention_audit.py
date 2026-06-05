"""Audit live CLOB trade-side convention against book transitions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lib.clob_book import ClobBook
from lib.trade_sign_normalization import historical_to_aggressor
from scripts.dali_clob_replay_features import as_float, parse_levels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "data" / "live_clob"
DEFAULT_OUT = ROOT / "notes" / "sign_convention_findings.md"
MIN_CLASSIFIED = 50
TOL = 1e-9


@dataclass
class PendingTrade:
    asset_id: str
    reported_side: str
    trade_price: float | None
    trade_size: float | None
    prior_bid: float | None
    prior_bid_size: float | None
    prior_ask: float | None
    prior_ask_size: float | None
    received_at: str
    transaction_hash: str


def classify_trade(pending: PendingTrade, book: ClobBook) -> str:
    top = book.top()
    if pending.trade_price is None:
        return "UNCLASSIFIABLE"

    if (
        pending.prior_ask is not None
        and abs(pending.trade_price - pending.prior_ask) <= TOL
        and pending.prior_ask_size is not None
        and top.ask_size is not None
        and top.ask_size < pending.prior_ask_size
    ):
        return "BUY"
    if (
        pending.prior_bid is not None
        and abs(pending.trade_price - pending.prior_bid) <= TOL
        and pending.prior_bid_size is not None
        and top.bid_size is not None
        and top.bid_size < pending.prior_bid_size
    ):
        return "SELL"
    if (
        pending.prior_bid is not None
        and pending.prior_ask is not None
        and pending.prior_bid < pending.trade_price < pending.prior_ask
    ):
        return "AMBIGUOUS"
    return "UNCLASSIFIABLE"


def update_book_from_message(book: ClobBook, event_type: str, msg: dict[str, Any]) -> None:
    if event_type == "book":
        book.replace(parse_levels(msg.get("bids")), parse_levels(msg.get("asks")))
    elif event_type == "price_change":
        price = as_float(msg.get("price"))
        size = as_float(msg.get("size"))
        side = str(msg.get("side") or "").upper()
        if price is not None and size is not None and side in {"BUY", "SELL"}:
            book.update_level(side, price, size)


def audit(paths: list[Path]) -> pd.DataFrame:
    books: dict[str, ClobBook] = {}
    pending: dict[str, list[PendingTrade]] = {}
    rows: list[dict[str, Any]] = []

    for path in paths:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                event_type = str(rec.get("event_type") or msg.get("event_type") or "")

                if event_type == "book":
                    asset_id = str(msg.get("asset_id") or "")
                    if not asset_id:
                        continue
                    book = books.setdefault(asset_id, ClobBook())
                    update_book_from_message(book, event_type, msg)
                    for trade in pending.pop(asset_id, []):
                        rows.append({**trade.__dict__, "inferred_aggressor": classify_trade(trade, book)})

                elif event_type == "price_change":
                    changes = msg.get("price_changes") or []
                    changes = [item for item in changes if isinstance(item, dict)]
                    for change in changes:
                        asset_id = str(change.get("asset_id") or "")
                        if not asset_id:
                            continue
                        book = books.setdefault(asset_id, ClobBook())
                        update_book_from_message(book, event_type, change)
                        if asset_id in pending:
                            for trade in pending.pop(asset_id):
                                rows.append({**trade.__dict__, "inferred_aggressor": classify_trade(trade, book)})

                elif event_type == "last_trade_price":
                    asset_id = str(msg.get("asset_id") or "")
                    if not asset_id:
                        continue
                    book = books.setdefault(asset_id, ClobBook())
                    top = book.top()
                    pending.setdefault(asset_id, []).append(
                        PendingTrade(
                            asset_id=asset_id,
                            reported_side=str(msg.get("side") or "").upper(),
                            trade_price=as_float(msg.get("price")),
                            trade_size=as_float(msg.get("size")),
                            prior_bid=top.bid_price,
                            prior_bid_size=top.bid_size,
                            prior_ask=top.ask_price,
                            prior_ask_size=top.ask_size,
                            received_at=str(rec.get("received_at") or ""),
                            transaction_hash=str(msg.get("transaction_hash") or ""),
                        )
                    )

    for trades in pending.values():
        for trade in trades:
            rows.append({**trade.__dict__, "inferred_aggressor": "UNCLASSIFIABLE"})
    return pd.DataFrame(rows)


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def small_markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if df.empty:
        return "No live trade rows found."
    rows = df[columns].head(limit).fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(row[col] for col in columns) + " |"
        for _, row in rows.iterrows()
    ]
    return "\n".join([header, sep, *body])


def write_note(df: pd.DataFrame, paths: list[Path], out: Path) -> None:
    total = len(df)
    classified = int(df["inferred_aggressor"].isin(["BUY", "SELL"]).sum()) if total else 0
    ambiguous = int(df["inferred_aggressor"].eq("AMBIGUOUS").sum()) if total else 0
    unclassifiable = int(df["inferred_aggressor"].eq("UNCLASSIFIABLE").sum()) if total else 0
    enough = classified >= MIN_CLASSIFIED

    p_buy_given_buy = "n/a"
    p_buy_given_sell = "n/a"
    if classified:
        buy_rows = df[df["inferred_aggressor"].eq("BUY")]
        sell_rows = df[df["inferred_aggressor"].eq("SELL")]
        p_buy_given_buy = pct(int(buy_rows["reported_side"].eq("BUY").sum()), len(buy_rows))
        p_buy_given_sell = pct(int(sell_rows["reported_side"].eq("BUY").sum()), len(sell_rows))

    source_list = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in paths)
    sample = ""
    if total:
        cols = [
            "received_at",
            "reported_side",
            "inferred_aggressor",
            "trade_price",
            "prior_bid",
            "prior_ask",
            "transaction_hash",
        ]
        sample = small_markdown_table(df, cols)

    status = (
        "Sign normalization is established from this sample."
        if enough
        else "Sign normalization is **not established** from this sample."
    )
    live_impl = (
        "`live_to_aggressor(..., semantics='aggressor')` may be used only if "
        "the classified-trade conditional probabilities support it."
        if enough
        else "`live_to_aggressor()` intentionally returns `UNKNOWN` by default "
        "until at least 50 classifiable live trades are available."
    )

    out.write_text(
        f"""# Dali Sign Convention Findings

Generated: 2026-05-23

## Sources

{source_list}

## Live CLOB Inference

- Total `last_trade_price` events inspected: {total}
- Classified from book transition: {classified} ({pct(classified, total)})
- Ambiguous: {ambiguous} ({pct(ambiguous, total)})
- Unclassifiable: {unclassifiable} ({pct(unclassifiable, total)})
- Minimum classifiable trades required to establish normalization: {MIN_CLASSIFIED}

{status}

Conditional checks if classified trades exist:

- `P(reported.side == BUY | inferred aggressor == BUY)`: {p_buy_given_buy}
- `P(reported.side == BUY | inferred aggressor == SELL)`: {p_buy_given_sell}

{live_impl}

## Historical Fill Semantics

Historical `maker_side` is the passive maker's token side. The aggressor is on
the opposite token side:

- `maker_side == BUY` means the maker bought tokens, so the taker/aggressor sold.
- `maker_side == SELL` means the maker sold tokens, so the taker/aggressor bought.

The helper `historical_to_aggressor()` implements this inversion. Current sanity
mapping:

- historical `BUY` -> `{historical_to_aggressor("BUY")}`
- historical `SELL` -> `{historical_to_aggressor("SELL")}`

## YES/NO Token Nuance

This audit is token-side, not market-direction-side. Buying YES and selling NO
can be economically similar for the underlying question, but they are different
CLOB token actions. Dali should keep token-side aggressor as the normalized
microstructure field, then map to market-direction only when outcome labels are
available and explicitly joined.

## Sample Rows

{sample or "No live trade rows found."}
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.latest:
        paths = sorted(DEFAULT_IN_DIR.glob("*.jsonl"))
    else:
        paths = args.jsonl
    if not paths:
        raise SystemExit("pass JSONL files or --latest")
    df = audit(paths)
    write_note(df, paths, args.out)
    print(f"audited {len(paths)} capture files")
    print(f"last_trade_price rows: {len(df)}")
    print(f"wrote {args.out.relative_to(ROOT)}")
    if len(df):
        print(df["inferred_aggressor"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
