"""Tests for signal/ — classifier covers entry/exit/scale, dedup drops repeats."""
from __future__ import annotations

import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    FillRecorded,
    JsonlWriter,
    LeaderFillObserved,
)
from polymarket.execution.signal import (
    Classifier,
    Deduplicator,
    MirrorSignal,
    Position,
    SignalKind,
    position_key,
)

_LEADER = "0x" + "a" * 40
_FUNDER = "0x" + "b" * 40

_BASE_CONFIG: dict[str, Any] = {
    "leader_address": _LEADER,
    "private_key": "0xpk",
    "api_key": "k",
    "api_secret": "s",
    "passphrase": "p",
    "funder": _FUNDER,
    "chain_id": 137,
    "signature_type": 1,
    "clob_url": "https://clob.polymarket.com",
    "gamma_url": "https://gamma-api.polymarket.com",
    "data_url": "https://data-api.polymarket.com",
    "ws_url": "wss://ws-live-data.polymarket.com",
    "max_capital_usd": 100.0,
    "per_trade_cap_usd": 20.0,
    "per_market_cap_usd": 50.0,
    "sizing_usd": 50.0,
    "max_open_positions": 3,
    "default_order_type": "FOK",
    "pricing_mode": "leader_fill",
    "price_deviation_pct": 2.0,
    "daily_loss_halt_usd": 200.0,
    "killswitch_path": Path("/tmp/polymarket_killswitch"),
    "journal_dir": Path("./journal_logs"),
    "log_level": "INFO",
    "max_real_orders": 5,
    "require_operator_confirm": False,
}


def _config(**overrides: Any) -> ExecutionConfig:
    return ExecutionConfig(**{**_BASE_CONFIG, **overrides})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fill(**overrides: Any) -> LeaderFillObserved:
    base = dict(
        ts_utc=_now(),
        transaction_hash="0xtx",
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        size=100.0,
        price=0.30,
        proxy_wallet=_LEADER,
        observed_at_utc=_now(),
    )
    base.update(overrides)
    return LeaderFillObserved(**base)


def _bot_fill(**overrides: Any) -> FillRecorded:
    base = dict(
        ts_utc=_now(),
        transaction_hash="0xbot",
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        size=10.0,
        price=0.50,
        proxy_wallet="0xbotwallet",
    )
    base.update(overrides)
    return FillRecorded(**base)


def _classifier(
    tmp_path: Path, **cfg_overrides: Any
) -> tuple[Classifier, JsonlWriter, Deduplicator, "queue.Queue[MirrorSignal]"]:
    journal = JsonlWriter(tmp_path, "sig-test")
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=100)
    cfg = _config(**cfg_overrides)
    classifier = Classifier(cfg, journal, dedup, sigq)
    return classifier, journal, dedup, sigq


def _events(journal: JsonlWriter) -> list[dict[str, Any]]:
    return list(journal.read_today())


# --- Dedup ----------------------------------------------------------

def test_first_fill_emits_signal(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill())
    journal.close()
    types = [e["event_type"] for e in _events(journal)]
    assert "LEADER_FILL_OBSERVED" not in types  # classifier doesn't journal these
    assert "MIRROR_SIGNAL_EMITTED" in types
    assert sigq.qsize() == 1


def test_duplicate_txhash_dropped(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xdup"))
    classifier.process_fill(_fill(transaction_hash="0xdup"))
    journal.close()
    events = _events(journal)
    drops = [e for e in events if e["event_type"] == "LEADER_FILL_DROPPED"]
    assert len(drops) == 1
    assert drops[0]["reason"] == "duplicate"
    signals = [e for e in events if e["event_type"] == "MIRROR_SIGNAL_EMITTED"]
    assert len(signals) == 1
    assert sigq.qsize() == 1


def test_dedup_state_survives_construction(tmp_path: Path) -> None:
    journal = JsonlWriter(tmp_path, "sig-test")
    journal.write(_fill(transaction_hash="0xseed"))
    journal.close()
    journal2 = JsonlWriter(tmp_path, "sig-test")
    dedup = Deduplicator(journal2)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal2, dedup, sigq)
    classifier.process_fill(_fill(transaction_hash="0xseed"))
    journal2.close()
    drops = [e for e in _events(journal2)
             if e["event_type"] == "LEADER_FILL_DROPPED"]
    assert any(d["reason"] == "duplicate" for d in drops)
    assert sigq.qsize() == 0


# --- ENTRY classification --------------------------------------------

def test_entry_target_size_from_sizing_usd(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(side="BUY", size=100.0, price=0.30))
    sig = sigq.get_nowait()
    assert sig.kind == SignalKind.ENTRY
    assert sig.target_size_shares == pytest.approx(50.0 / 0.30)


def test_entry_scale_up_emits_two_signals(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xa", size=100.0, price=0.30))
    classifier.process_fill(_fill(transaction_hash="0xb", size=50.0, price=0.40))
    assert sigq.qsize() == 2
    leader_pos = classifier._leader_positions[position_key("0xcond", "42")]
    assert leader_pos.shares == 150.0


def test_entry_two_markets_tracked_separately(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xa", condition_id="0xA",
                                  asset_id="1", size=100.0, price=0.30))
    classifier.process_fill(_fill(transaction_hash="0xb", condition_id="0xB",
                                  asset_id="2", size=200.0, price=0.50))
    assert sigq.qsize() == 2
    assert position_key("0xA", "1") in classifier._leader_positions
    assert position_key("0xB", "2") in classifier._leader_positions


# --- EXIT classification ---------------------------------------------

def _seed_bot_and_leader(
    tmp_path: Path,
    bot_shares: float,
    leader_shares: float,
    *,
    condition_id: str = "0xcond",
    asset_id: str = "42",
    price: float = 0.30,
) -> JsonlWriter:
    journal = JsonlWriter(tmp_path, "sig-test")
    journal.write(_bot_fill(transaction_hash="0xbot", condition_id=condition_id,
                            asset_id=asset_id, side="BUY",
                            size=bot_shares, price=price))
    journal.write(_fill(transaction_hash="0xleader", condition_id=condition_id,
                        asset_id=asset_id, side="BUY",
                        size=leader_shares, price=price))
    return journal


def test_exit_partial_proportional(tmp_path: Path) -> None:
    journal = _seed_bot_and_leader(tmp_path, bot_shares=50.0, leader_shares=1000.0)
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal, dedup, sigq)
    classifier.process_fill(_fill(transaction_hash="0xsell",
                                  side="SELL", size=300.0, price=0.30))
    sig = sigq.get_nowait()
    assert sig.kind == SignalKind.EXIT
    assert sig.target_size_shares == pytest.approx(0.3 * 50.0)


def test_exit_full(tmp_path: Path) -> None:
    journal = _seed_bot_and_leader(tmp_path, bot_shares=50.0, leader_shares=1000.0)
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal, dedup, sigq)
    classifier.process_fill(_fill(transaction_hash="0xfull",
                                  side="SELL", size=1000.0, price=0.30))
    sig = sigq.get_nowait()
    assert sig.kind == SignalKind.EXIT
    assert sig.target_size_shares == pytest.approx(50.0)


def test_exit_bot_no_position(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    # Seed leader only via process_fill BUY first.
    classifier.process_fill(_fill(transaction_hash="0xb", side="BUY",
                                  size=100.0, price=0.30))
    sigq.get_nowait()  # drain the ENTRY signal
    classifier.process_fill(_fill(transaction_hash="0xs", side="SELL",
                                  size=50.0, price=0.30))
    journal.close()
    drops = [e for e in _events(journal)
             if e["event_type"] == "LEADER_FILL_DROPPED"]
    assert any(d["reason"] == "no_position" for d in drops)
    assert sigq.qsize() == 0
    # Leader state still updated.
    assert classifier._leader_positions[position_key("0xcond", "42")].shares == 50.0


def test_exit_leader_no_position(tmp_path: Path) -> None:
    classifier, journal, _dedup, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xs", side="SELL",
                                  size=10.0, price=0.30))
    journal.close()
    drops = [e for e in _events(journal)
             if e["event_type"] == "LEADER_FILL_DROPPED"]
    assert any(d["reason"] == "leader_no_position" for d in drops)
    assert sigq.qsize() == 0


# --- State rebuild ---------------------------------------------------

def test_rebuild_empty_journal(tmp_path: Path) -> None:
    classifier, _journal, _dedup, _sigq = _classifier(tmp_path)
    assert classifier._bot_positions == {}
    assert classifier._leader_positions == {}


def test_rebuild_bot_positions_from_fills(tmp_path: Path) -> None:
    journal = JsonlWriter(tmp_path, "sig-test")
    journal.write(_bot_fill(transaction_hash="0xa", side="BUY", size=10.0, price=0.50))
    journal.write(_bot_fill(transaction_hash="0xb", side="BUY", size=20.0, price=0.50))
    journal.write(_bot_fill(transaction_hash="0xc", side="SELL", size=5.0, price=0.50))
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal, dedup, sigq)
    pos = classifier._bot_positions[position_key("0xcond", "42")]
    assert pos.shares == 25.0


def test_rebuild_leader_positions_from_observed(tmp_path: Path) -> None:
    journal = JsonlWriter(tmp_path, "sig-test")
    journal.write(_fill(transaction_hash="0xa", side="BUY", size=10.0, price=0.40))
    journal.write(_fill(transaction_hash="0xb", side="BUY", size=20.0, price=0.50))
    journal.write(_fill(transaction_hash="0xc", side="SELL", size=5.0, price=0.50))
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal, dedup, sigq)
    pos = classifier._leader_positions[position_key("0xcond", "42")]
    assert pos.shares == 25.0


# --- Position math --------------------------------------------------

def test_position_math_share_weighted_average(tmp_path: Path) -> None:
    classifier, _j, _d, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xa", size=10.0, price=0.40))
    classifier.process_fill(_fill(transaction_hash="0xb", size=20.0, price=0.50))
    pos = classifier._leader_positions[position_key("0xcond", "42")]
    assert pos.shares == 30.0
    assert pos.avg_entry_price == pytest.approx((10 * 0.40 + 20 * 0.50) / 30)
    assert pos.total_entry_usd == pytest.approx(14.0)


def test_position_math_partial_sell_preserves_avg(tmp_path: Path) -> None:
    classifier, _j, _d, sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xa", side="BUY",
                                  size=10.0, price=0.50))
    # leader has 10 shares now; sell 5
    classifier.process_fill(_fill(transaction_hash="0xb", side="SELL",
                                  size=5.0, price=0.50))
    pos = classifier._leader_positions[position_key("0xcond", "42")]
    assert pos.shares == 5.0
    assert pos.avg_entry_price == pytest.approx(0.50)
    assert pos.total_entry_usd == pytest.approx(2.50)


def test_position_math_oversell_clamps_to_zero(tmp_path: Path) -> None:
    classifier, _j, _d, _sigq = _classifier(tmp_path)
    classifier.process_fill(_fill(transaction_hash="0xa", side="BUY",
                                  size=10.0, price=0.50))
    classifier.process_fill(_fill(transaction_hash="0xb", side="SELL",
                                  size=999.0, price=0.50))
    assert position_key("0xcond", "42") not in classifier._leader_positions


# --- Pipeline ordering ----------------------------------------------

def test_dedup_blocks_double_position_update(tmp_path: Path) -> None:
    journal = _seed_bot_and_leader(tmp_path, bot_shares=50.0, leader_shares=100.0)
    dedup = Deduplicator(journal)
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    classifier = Classifier(_config(), journal, dedup, sigq)
    # First EXIT fill — leader 100 → 50, bot signal emitted.
    classifier.process_fill(_fill(transaction_hash="0xexit",
                                  side="SELL", size=50.0, price=0.30))
    leader_after_first = classifier._leader_positions[
        position_key("0xcond", "42")
    ].shares
    assert leader_after_first == 50.0
    # Same txhash again — must be a no-op for state.
    classifier.process_fill(_fill(transaction_hash="0xexit",
                                  side="SELL", size=50.0, price=0.30))
    leader_after_dup = classifier._leader_positions[
        position_key("0xcond", "42")
    ].shares
    assert leader_after_dup == 50.0  # unchanged
    assert sigq.qsize() == 1  # only the first signal queued
