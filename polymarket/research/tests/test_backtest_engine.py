from __future__ import annotations

import json

from lib.backtest_engine import BacktestConfig, BacktestEngine


def write_jsonl(path, records) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_backtest_executes_at_executable_price_and_takes_profit(tmp_path) -> None:
    path = tmp_path / "capture.jsonl"
    records = [
        {
            "received_at": "2026-05-23T00:00:00.000Z",
            "event_type": "book",
            "message": {
                "event_type": "book",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400000",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.51", "size": "100"}],
            },
        },
        {
            "received_at": "2026-05-23T00:00:00.100Z",
            "event_type": "last_trade_price",
            "message": {
                "event_type": "last_trade_price",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400100",
                "price": "0.51",
                "side": "BUY",
                "size": "5",
                "transaction_hash": "0x1",
            },
        },
        {
            "received_at": "2026-05-23T00:00:00.200Z",
            "event_type": "best_bid_ask",
            "message": {
                "event_type": "best_bid_ask",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400200",
                "best_bid": "0.50",
                "best_ask": "0.52",
            },
        },
        {
            "received_at": "2026-05-23T00:00:01.000Z",
            "event_type": "book",
            "message": {
                "event_type": "book",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494401000",
                "bids": [{"price": "0.53", "size": "100"}],
                "asks": [{"price": "0.54", "size": "100"}],
            },
        },
    ]
    write_jsonl(path, records)

    engine = BacktestEngine(
        BacktestConfig(
            fixed_latency_ms=0,
            order_size=5,
            tick_size=0.01,
            take_profit_ticks=1,
            stop_loss_ticks=5,
            force_close_at_end=True,
        )
    )
    journal = engine.run_jsonl(path)

    assert len(journal) == 1
    trade = journal.iloc[0]
    assert trade["entry_price"] == 0.51
    assert trade["exit_price"] == 0.53
    assert trade["exit_reason"] == "take_profit"
    assert trade["gross_pnl"] == (0.53 - 0.51) * 5
    assert trade["net_pnl"] == trade["gross_pnl"]


def test_backtest_walks_book_for_partial_fill(tmp_path) -> None:
    path = tmp_path / "capture.jsonl"
    records = [
        {
            "received_at": "2026-05-23T00:00:00.000Z",
            "event_type": "book",
            "message": {
                "event_type": "book",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400000",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [
                    {"price": "0.51", "size": "2"},
                    {"price": "0.52", "size": "3"},
                ],
            },
        },
        {
            "received_at": "2026-05-23T00:00:00.100Z",
            "event_type": "last_trade_price",
            "message": {
                "event_type": "last_trade_price",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400100",
                "price": "0.51",
                "side": "BUY",
                "size": "5",
                "transaction_hash": "0x1",
            },
        },
        {
            "received_at": "2026-05-23T00:00:01.000Z",
            "event_type": "book",
            "message": {
                "event_type": "book",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494401000",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.52", "size": "100"}],
            },
        },
    ]
    write_jsonl(path, records)
    engine = BacktestEngine(
        BacktestConfig(
            fixed_latency_ms=0,
            order_size=10,
            tick_size=0.01,
            take_profit_ticks=99,
            stop_loss_ticks=99,
            force_close_at_end=True,
        )
    )
    journal = engine.run_jsonl(path)

    assert len(journal) == 1
    trade = journal.iloc[0]
    assert trade["partial_fill"]
    assert trade["size"] == 5
    assert trade["entry_price"] == ((0.51 * 2) + (0.52 * 3)) / 5
