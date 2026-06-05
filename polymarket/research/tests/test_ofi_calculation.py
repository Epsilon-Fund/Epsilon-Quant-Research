from __future__ import annotations

import json
from importlib import import_module

import pytest

from lib.clob_book import ClobBook


def test_ofi_top_bid_depth_increase() -> None:
    book = ClobBook()
    assert book.replace([(0.50, 100.0)], [(0.51, 100.0)]).combined == 0.0

    ofi = book.update_level("BUY", 0.50, 150.0)

    assert ofi.bid == pytest.approx(50.0)
    assert ofi.ask == pytest.approx(0.0)
    assert ofi.combined == pytest.approx(50.0)


def test_ofi_top_bid_price_increase() -> None:
    book = ClobBook()
    book.replace([(0.50, 100.0)], [(0.53, 100.0)])

    ofi = book.update_level("BUY", 0.51, 80.0)

    assert ofi.bid == pytest.approx(80.0)
    assert ofi.ask == pytest.approx(0.0)
    assert ofi.combined == pytest.approx(80.0)


def test_ofi_top_bid_price_decrease() -> None:
    book = ClobBook()
    book.replace([(0.50, 100.0), (0.49, 40.0)], [(0.53, 100.0)])

    ofi = book.update_level("BUY", 0.50, 0.0)

    assert ofi.bid == pytest.approx(-100.0)
    assert ofi.ask == pytest.approx(0.0)
    assert ofi.combined == pytest.approx(-100.0)


def test_ofi_top_ask_price_decrease_has_negative_sign() -> None:
    book = ClobBook()
    book.replace([(0.48, 100.0)], [(0.51, 100.0)])

    ofi = book.update_level("SELL", 0.50, 80.0)

    assert ofi.bid == pytest.approx(0.0)
    assert ofi.ask == pytest.approx(-80.0)
    assert ofi.combined == pytest.approx(-80.0)


def test_multi_level_ofi_tracks_unchanged_deeper_bid_size() -> None:
    book = ClobBook()
    first = book.multi_level_replace(
        [(0.50, 100.0), (0.49, 40.0)],
        [(0.51, 90.0), (0.52, 30.0)],
        depth=3,
    )
    assert first.combined == pytest.approx((0.0, 0.0, 0.0))

    ofi = book.multi_level_update_level("BUY", 0.49, 70.0, depth=3)

    assert ofi.bid == pytest.approx((0.0, 30.0, 0.0))
    assert ofi.ask == pytest.approx((0.0, 0.0, 0.0))
    assert ofi.combined == pytest.approx((0.0, 30.0, 0.0))


def test_multi_level_ofi_tracks_unchanged_deeper_ask_size() -> None:
    book = ClobBook()
    book.multi_level_replace(
        [(0.50, 100.0), (0.49, 40.0)],
        [(0.51, 90.0), (0.52, 30.0)],
        depth=3,
    )

    ofi = book.multi_level_update_level("SELL", 0.52, 10.0, depth=3)

    assert ofi.bid == pytest.approx((0.0, 0.0, 0.0))
    assert ofi.ask == pytest.approx((0.0, 20.0, 0.0))
    assert ofi.combined == pytest.approx((0.0, 20.0, 0.0))


def test_best_bid_ask_does_not_mutate_executable_book(tmp_path) -> None:
    replay_mod = import_module("scripts.dali_clob_replay_features")
    path = tmp_path / "sample.jsonl"
    records = [
        {
            "received_at": "2026-05-23T00:00:00.000Z",
            "event_type": "book",
            "message": {
                "event_type": "book",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494400000",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.60", "size": "100"}],
            },
        },
        {
            "received_at": "2026-05-23T00:00:01.000Z",
            "event_type": "best_bid_ask",
            "message": {
                "event_type": "best_bid_ask",
                "asset_id": "asset-1",
                "market": "market-1",
                "timestamp": "1779494401000",
                "best_bid": "0.55",
                "best_ask": "0.56",
                "spread": "0.01",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    df = replay_mod.replay(path, top_n=1)
    best_row = df[df["event_type"].eq("best_bid_ask")].iloc[0]

    assert best_row["best_bid"] == pytest.approx(0.50)
    assert best_row["best_ask"] == pytest.approx(0.60)
    assert best_row["telemetry_best_bid"] == pytest.approx(0.55)
    assert best_row["telemetry_best_ask"] == pytest.approx(0.56)


def test_a1_future_mid_uses_real_seconds_with_microsecond_timestamps() -> None:
    a1_mod = import_module("scripts.dali_block_a1_analyze")
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "received_at": pd.to_datetime(
                [
                    "2026-05-28T00:00:00.000001Z",
                    "2026-05-28T00:00:01.000001Z",
                    "2026-05-28T00:05:00.000001Z",
                ],
                utc=True,
            ),
            "mid": [0.50, 0.51, 0.60],
        }
    )

    one_second = a1_mod.future_mid(df, 1)
    five_minutes = a1_mod.future_mid(df, 300)

    assert one_second[0] == pytest.approx(0.51)
    assert five_minutes[0] == pytest.approx(0.60)


def test_a1_latency_slippage_uses_milliseconds_not_100_seconds() -> None:
    a1_mod = import_module("scripts.dali_block_a1_analyze")
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "received_at": pd.to_datetime(
                [
                    "2026-05-28T00:00:00.000000Z",
                    "2026-05-28T00:00:00.100000Z",
                    "2026-05-28T00:00:00.200000Z",
                    "2026-05-28T00:00:00.300000Z",
                    "2026-05-28T00:00:00.400000Z",
                    "2026-05-28T00:00:00.500000Z",
                    "2026-05-28T00:01:40.000000Z",
                    "2026-05-28T00:01:40.100000Z",
                    "2026-05-28T00:01:40.200000Z",
                    "2026-05-28T00:01:40.300000Z",
                    "2026-05-28T00:01:40.400000Z",
                    "2026-05-28T00:01:40.500000Z",
                ],
                utc=True,
            ),
            "directional_mid": [
                0.500,
                0.501,
                0.502,
                0.503,
                0.504,
                0.505,
                0.800,
                0.801,
                0.802,
                0.803,
                0.804,
                0.805,
            ],
        }
    )

    slippage = a1_mod.latency_slippage_bps(df, 100)

    assert slippage < 25.0


def test_a1_taker_fee_bps_matches_return_denominator() -> None:
    a1_mod = import_module("scripts.dali_block_a1_analyze")

    token_notional_bps = a1_mod.taker_fee_bps("Crypto", 0.40)
    directional_return_bps = a1_mod.taker_fee_bps("Crypto", 0.40, 0.60)

    assert token_notional_bps == pytest.approx(420.0)
    assert directional_return_bps == pytest.approx(280.0)


def test_a1_reportable_guard_requires_enough_top_decile_rows() -> None:
    a1_mod = import_module("scripts.dali_block_a1_analyze")

    thin_top_decile = a1_mod.MetricResult(
        r2=0.01,
        r2_lo=0.0,
        r2_hi=0.02,
        hit_rate=1.0,
        hit_lo=1.0,
        hit_hi=1.0,
        directional_return_bps=10.0,
        n_eval=100,
        top_decile_n=1,
    )
    usable_top_decile = a1_mod.MetricResult(
        r2=0.01,
        r2_lo=0.0,
        r2_hi=0.02,
        hit_rate=0.60,
        hit_lo=0.55,
        hit_hi=0.65,
        directional_return_bps=10.0,
        n_eval=100,
        top_decile_n=30,
    )
    perfect_hit_top_decile = a1_mod.MetricResult(
        r2=0.01,
        r2_lo=0.0,
        r2_hi=0.02,
        hit_rate=1.0,
        hit_lo=1.0,
        hit_hi=1.0,
        directional_return_bps=10.0,
        n_eval=100,
        top_decile_n=100,
    )

    assert not a1_mod.metric_reportable(200, thin_top_decile)
    assert not a1_mod.metric_reportable(29, usable_top_decile)
    assert not a1_mod.metric_reportable(200, perfect_hit_top_decile)
    assert a1_mod.metric_reportable(200, usable_top_decile)
