"""Tests for mirror/mirror_engine.py."""
from __future__ import annotations

import queue
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import FillRecorded, JsonlWriter
from polymarket.execution.mirror import MirrorEngine, SubmitResult
from polymarket.execution.mirror import mirror_engine as me_mod
from polymarket.execution.signal import MirrorSignal, SignalKind, position_key

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
    "per_trade_cap_usd": 60.0,
    "per_market_cap_usd": 80.0,
    "sizing_usd": 50.0,
    "max_open_positions": 3,
    "default_order_type": "FOK",
    "pricing_mode": "current_book",
    "price_deviation_pct": 2.0,
    "daily_loss_halt_usd": 200.0,
    "killswitch_path": Path("/tmp/polymarket_killswitch_unused"),
    "journal_dir": Path("./journal_logs"),
    "log_level": "INFO",
    "max_real_orders": 5,
    "require_operator_confirm": False,
}


def _config(**overrides: Any) -> ExecutionConfig:
    return ExecutionConfig(**{**_BASE_CONFIG, **overrides})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _signal(**overrides: Any) -> MirrorSignal:
    base = dict(
        signal_id=uuid.uuid4().hex,
        kind=SignalKind.ENTRY,
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        target_size_shares=100.0,
        leader_fill_price=0.40,
        source_transaction_hash="0xtx",
    )
    base.update(overrides)
    return MirrorSignal(**base)


class _FakeVenue:
    def __init__(
        self,
        *,
        raise_exc: BaseException | None = None,
        accepted: bool = True,
        ambiguous: bool = False,
        message: str | None = None,
        immediate_fill: bool = True,
        real: bool = False,
    ) -> None:
        self._raise = raise_exc
        self._accepted = accepted
        self._ambiguous = ambiguous
        self._message = message
        self._immediate_fill = immediate_fill
        self._real = real
        self.calls: list[dict[str, Any]] = []

    def is_real_venue(self) -> bool:
        return self._real

    def submit_order(self, **kwargs: Any) -> SubmitResult:
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        if self._ambiguous or not self._accepted:
            return SubmitResult(
                accepted=self._accepted,
                ambiguous=self._ambiguous,
                venue_order_id=None,
                message=self._message,
            )
        if self._immediate_fill:
            return SubmitResult(
                accepted=True,
                ambiguous=False,
                venue_order_id=f"venue-{kwargs['client_order_id']}",
                fill_price=kwargs["price"],
                fill_size_shares=kwargs["size_shares"],
            )
        return SubmitResult(
            accepted=True,
            ambiguous=False,
            venue_order_id=f"venue-{kwargs['client_order_id']}",
        )


@pytest.fixture(autouse=True)
def _stub_orderbook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: orderbook returns a sensible price. Tests can override."""
    monkeypatch.setattr(me_mod.orderbook, "get_best_price",
                        lambda url, asset, side: 0.40)


def _engine(
    tmp_path: Path,
    *,
    venue: _FakeVenue | None = None,
    config: ExecutionConfig | None = None,
    kill_switch_path: Path | None = None,
    seed_fills: list[FillRecorded] | None = None,
) -> tuple[MirrorEngine, JsonlWriter, _FakeVenue, "queue.Queue[MirrorSignal]"]:
    journal = JsonlWriter(tmp_path, "mirror-test")
    if seed_fills:
        for f in seed_fills:
            journal.write(f)
    cfg = config or _config()
    v = venue or _FakeVenue()
    sigq: queue.Queue[MirrorSignal] = queue.Queue(maxsize=10)
    ks = kill_switch_path or (tmp_path / "killswitch")
    engine = MirrorEngine(cfg, journal, v, sigq, ks)
    return engine, journal, v, sigq


def _events(journal: JsonlWriter) -> list[dict[str, Any]]:
    return list(journal.read_today())


def _fill(**overrides: Any) -> FillRecorded:
    base = dict(
        ts_utc=_now(),
        transaction_hash=f"0x{uuid.uuid4().hex[:16]}",
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        size=100.0,
        price=0.40,
        proxy_wallet=_FUNDER,
    )
    base.update(overrides)
    return FillRecorded(**base)


# --- Happy path -----------------------------------------------------

def test_entry_signal_submits_and_records_fill(tmp_path: Path) -> None:
    engine, journal, venue, _q = _engine(tmp_path)
    engine.handle_signal(_signal(kind=SignalKind.ENTRY, side="BUY"))
    journal.close()
    types = [e["event_type"] for e in _events(journal)]
    assert "ORDER_SUBMITTED" in types
    assert "ORDER_ACKNOWLEDGED" in types
    assert "FILL_RECORDED" in types
    pos = engine._bot_positions[position_key("0xcond", "42")]
    assert pos.shares > 0
    assert len(venue.calls) == 1
    assert venue.calls[0]["side"] == "BUY"


def test_exit_signal_decrements_position(tmp_path: Path) -> None:
    seed = [_fill(side="BUY", size=50.0, price=0.40,
                  transaction_hash="0xseed")]
    engine, journal, venue, _q = _engine(tmp_path, seed_fills=seed)
    # Construct should have set the position to 50.
    assert engine._bot_positions[position_key("0xcond", "42")].shares == 50.0
    engine.handle_signal(_signal(
        kind=SignalKind.EXIT, side="SELL", target_size_shares=25.0,
        leader_fill_price=0.40,
    ))
    pos = engine._bot_positions.get(position_key("0xcond", "42"))
    assert pos is not None
    # EXITs are share-sized. The submitted SELL must be exactly the
    # signal's target_size_shares (within 4-decimal rounding).
    assert venue.calls[0]["side"] == "SELL"
    assert venue.calls[0]["size_shares"] == pytest.approx(25.0, abs=1e-4)
    assert pos.shares == pytest.approx(25.0, abs=1e-4)


def test_exit_share_count_unchanged_when_price_diverges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXIT submits exactly target_size_shares, regardless of how far
    current price has drifted from leader_fill_price. Locks in the
    fix for the share/USD round-trip bug."""
    monkeypatch.setattr(me_mod.orderbook, "get_best_price",
                        lambda url, asset, side: 0.30)  # current
    # Widen price_deviation tolerance so the breaker doesn't block
    # this test (the 28% gap between 0.30 and 0.42 is the *point* —
    # we want to verify share-count behaviour with a real gap).
    cfg = _config(price_deviation_pct=50.0)
    seed = [_fill(side="BUY", size=50.0, price=0.42,
                  transaction_hash="0xseed")]
    engine, _journal, venue, _q = _engine(tmp_path, config=cfg, seed_fills=seed)
    engine.handle_signal(_signal(
        kind=SignalKind.EXIT, side="SELL", target_size_shares=15.0,
        leader_fill_price=0.42,
    ))
    assert venue.calls[0]["size_shares"] == pytest.approx(15.0, abs=1e-4)


def test_entry_target_shares_from_sizing_usd_and_submission_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENTRY: size_usd is fixed at config.sizing_usd; target_shares
    is derived as sizing_usd / submission_price. Locks in the
    dollar-sized semantics for entries."""
    monkeypatch.setattr(me_mod.orderbook, "get_best_price",
                        lambda url, asset, side: 0.40)
    cfg = _config(sizing_usd=50.0, price_deviation_pct=2.0,
                  per_trade_cap_usd=200.0, per_market_cap_usd=200.0,
                  max_capital_usd=1000.0)
    engine, _journal, venue, _q = _engine(tmp_path, config=cfg)
    engine.handle_signal(_signal(
        kind=SignalKind.ENTRY, side="BUY",
        target_size_shares=999.0,  # ignored for ENTRY
        leader_fill_price=0.40,
    ))
    # submission_price = round(0.40 * 1.02, 2) = 0.41
    # target_shares  = round(50.0 / 0.41, 4) = 121.9512
    assert venue.calls[0]["price"] == pytest.approx(0.41, abs=1e-9)
    assert venue.calls[0]["size_shares"] == pytest.approx(50.0 / 0.41, abs=1e-4)


# --- Risk vetos (skip-and-continue) ---------------------------------

def test_size_cap_veto_no_submission(tmp_path: Path) -> None:
    cfg = _config(per_trade_cap_usd=10.0, sizing_usd=50.0)
    engine, journal, venue, _q = _engine(tmp_path, config=cfg)
    engine.handle_signal(_signal())
    journal.close()
    risk_halts = [e for e in _events(journal) if e["event_type"] == "RISK_HALT"]
    assert any(e["reason"] == "size_cap" for e in risk_halts)
    assert venue.calls == []
    assert engine.is_halted() is False


def test_market_cap_veto_no_submission(tmp_path: Path) -> None:
    seed = [_fill(side="BUY", size=70.0, price=1.00,
                  transaction_hash="0xseed")]  # $70 already in market
    cfg = _config(per_market_cap_usd=80.0, sizing_usd=50.0,
                  per_trade_cap_usd=60.0, max_capital_usd=200.0)
    engine, journal, venue, _q = _engine(tmp_path, config=cfg, seed_fills=seed)
    engine.handle_signal(_signal())
    journal.close()
    risk_halts = [e for e in _events(journal) if e["event_type"] == "RISK_HALT"]
    assert any(e["reason"] == "market_cap" for e in risk_halts)
    assert venue.calls == []
    assert engine.is_halted() is False


def test_daily_loss_veto_no_submission(tmp_path: Path) -> None:
    # Realised PnL: -100*1.0 + 0*0 = -100; just below -50 halt threshold.
    seed = [
        _fill(side="BUY", size=100.0, price=1.00,
              transaction_hash="0xb1", condition_id="0xother", asset_id="9"),
    ]
    cfg = _config(daily_loss_halt_usd=50.0, max_capital_usd=10_000.0,
                  per_trade_cap_usd=10_000.0, per_market_cap_usd=10_000.0)
    engine, journal, venue, _q = _engine(tmp_path, config=cfg, seed_fills=seed)
    assert engine._daily_realised_pnl_usd <= -50.0
    engine.handle_signal(_signal())
    journal.close()
    risk_halts = [e for e in _events(journal) if e["event_type"] == "RISK_HALT"]
    assert any(e["reason"] == "daily_loss" for e in risk_halts)
    assert venue.calls == []
    assert engine.is_halted() is False


# --- Kill switch (only veto that halts) -----------------------------

def test_kill_switch_present_halts(tmp_path: Path) -> None:
    ks = tmp_path / "killswitch"
    ks.write_text("HALT")
    engine, journal, venue, _q = _engine(tmp_path, kill_switch_path=ks)
    engine.handle_signal(_signal())
    journal.close()
    risk_halts = [e for e in _events(journal) if e["event_type"] == "RISK_HALT"]
    assert any(e["reason"] == "kill_switch" for e in risk_halts)
    assert engine.is_halted() is True
    assert venue.calls == []


# --- Submission failures --------------------------------------------

def test_venue_exception_halts(tmp_path: Path) -> None:
    venue = _FakeVenue(raise_exc=ConnectionError("boom"))
    engine, journal, _v, _q = _engine(tmp_path, venue=venue)
    engine.handle_signal(_signal())
    journal.close()
    rejects = [e for e in _events(journal) if e["event_type"] == "ORDER_REJECTED"]
    assert any(r["reason"] == "submit_exception" for r in rejects)
    assert engine.is_halted() is True


def test_ambiguous_submit_halts(tmp_path: Path) -> None:
    venue = _FakeVenue(ambiguous=True, message="timeout")
    engine, journal, _v, _q = _engine(tmp_path, venue=venue)
    engine.handle_signal(_signal())
    journal.close()
    types = [e["event_type"] for e in _events(journal)]
    assert "AMBIGUOUS_SUBMIT" in types
    assert engine.is_halted() is True


def test_venue_rejection_does_not_halt(tmp_path: Path) -> None:
    venue = _FakeVenue(accepted=False, message="FOK rejected")
    engine, journal, _v, _q = _engine(tmp_path, venue=venue)
    engine.handle_signal(_signal())
    journal.close()
    rejects = [e for e in _events(journal) if e["event_type"] == "ORDER_REJECTED"]
    assert any(r["reason"] == "venue_rejected" for r in rejects)
    assert engine.is_halted() is False


# --- State rebuild --------------------------------------------------

def test_state_rebuild_from_journal(tmp_path: Path) -> None:
    seed = [
        _fill(side="BUY", size=100.0, price=0.40, transaction_hash="0xa"),
        _fill(side="SELL", size=50.0, price=0.60, transaction_hash="0xb"),
    ]
    engine, _journal, _v, _q = _engine(tmp_path, seed_fills=seed)
    pos = engine._bot_positions[position_key("0xcond", "42")]
    assert pos.shares == 50.0
    assert pos.avg_entry_price == pytest.approx(0.40)
    # Realised PnL: -100*0.40 + 50*0.60 = -40 + 30 = -10.
    assert engine._daily_realised_pnl_usd == pytest.approx(-10.0)


# --- Orderbook fetch failure ----------------------------------------

def test_no_orderbook_price_drops_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(me_mod.orderbook, "get_best_price",
                        lambda url, asset, side: None)
    engine, journal, venue, _q = _engine(tmp_path)
    engine.handle_signal(_signal())
    journal.close()
    rejects = [e for e in _events(journal) if e["event_type"] == "ORDER_REJECTED"]
    assert any(r["reason"] == "no_orderbook_price" for r in rejects)
    assert venue.calls == []
    assert engine.is_halted() is False


# --- Halt is sticky -------------------------------------------------

# --- pricing_mode dispatch (leader_fill vs current_book) -------

@pytest.mark.parametrize("kind,side", [
    (SignalKind.ENTRY, "BUY"),
    (SignalKind.EXIT, "SELL"),
])
def test_leader_fill_mode_uses_leader_price_no_orderbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    kind: SignalKind, side: str,
) -> None:
    """leader_fill mode: submission_price = signal.leader_fill_price.
    Orderbook helper is NOT consulted — even if it would explode."""
    fetch_calls: list[tuple] = []

    def explode(*args, **kwargs):
        fetch_calls.append(args)
        raise RuntimeError("orderbook should not be consulted in leader_fill mode")

    monkeypatch.setattr(me_mod.orderbook, "get_best_price", explode)

    cfg = _config(pricing_mode="leader_fill")
    seed = (
        [_fill(side="BUY", size=50.0, price=0.42, transaction_hash="0xseed")]
        if kind == SignalKind.EXIT else None
    )
    engine, _journal, venue, _q = _engine(tmp_path, config=cfg, seed_fills=seed)
    engine.handle_signal(_signal(
        kind=kind, side=side, target_size_shares=15.0, leader_fill_price=0.42,
    ))
    assert fetch_calls == []  # orderbook never consulted
    assert len(venue.calls) == 1
    assert venue.calls[0]["price"] == pytest.approx(0.42, abs=1e-9)


@pytest.mark.parametrize("kind,side,want_price", [
    # ENTRY BUY:  0.40 * (1 + 0.02) → 0.408 → round(0.41)
    (SignalKind.ENTRY, "BUY", 0.41),
    # EXIT SELL:  0.40 * (1 - 0.02) → 0.392 → round(0.39)
    (SignalKind.EXIT, "SELL", 0.39),
])
def test_current_book_mode_fetches_orderbook_and_applies_slippage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    kind: SignalKind, side: str, want_price: float,
) -> None:
    fetch_calls: list[tuple] = []

    def fake_fetch(url, asset, side_):
        fetch_calls.append((url, asset, side_))
        return 0.40

    monkeypatch.setattr(me_mod.orderbook, "get_best_price", fake_fetch)

    cfg = _config(pricing_mode="current_book", price_deviation_pct=2.0)
    seed = (
        [_fill(side="BUY", size=50.0, price=0.40, transaction_hash="0xseed")]
        if kind == SignalKind.EXIT else None
    )
    engine, _journal, venue, _q = _engine(tmp_path, config=cfg, seed_fills=seed)
    # leader_fill_price matches current to keep price_deviation breaker happy.
    engine.handle_signal(_signal(
        kind=kind, side=side, target_size_shares=15.0, leader_fill_price=0.40,
    ))
    assert len(fetch_calls) == 1
    assert len(venue.calls) == 1
    assert venue.calls[0]["price"] == pytest.approx(want_price, abs=1e-9)


def test_leader_fill_mode_tolerates_orderbook_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In leader_fill mode, an orderbook RuntimeError must not affect
    the submission path (the helper isn't called at all)."""
    monkeypatch.setattr(
        me_mod.orderbook, "get_best_price",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CLOB down")),
    )
    cfg = _config(pricing_mode="leader_fill")
    engine, journal, venue, _q = _engine(tmp_path, config=cfg)
    engine.handle_signal(_signal(
        kind=SignalKind.ENTRY, side="BUY", leader_fill_price=0.42,
    ))
    journal.close()
    types = [e["event_type"] for e in _events(journal)]
    assert "ORDER_REJECTED" not in types
    assert "ORDER_ACKNOWLEDGED" in types
    assert venue.calls[0]["price"] == pytest.approx(0.42, abs=1e-9)


def test_halted_engine_skips_subsequent_signals(tmp_path: Path) -> None:
    venue = _FakeVenue(ambiguous=True)  # first signal halts
    engine, journal, _v, _q = _engine(tmp_path, venue=venue)
    engine.handle_signal(_signal())
    assert engine.is_halted() is True
    # Second call: venue should NOT be invoked.
    venue_calls_before = len(venue.calls)
    engine.handle_signal(_signal())
    assert len(venue.calls) == venue_calls_before


# ---------------------------------------------------------------------------
# Real-venue safety harness
# ---------------------------------------------------------------------------


def test_max_real_orders_halts_after_n_submits(tmp_path: Path) -> None:
    venue = _FakeVenue(real=True, immediate_fill=False)
    # Wide caps so risk doesn't veto before the harness fires; distinct
    # condition_ids per signal so per_market_cap doesn't accumulate.
    cfg = _config(
        max_real_orders=2, per_trade_cap_usd=1000.0,
        per_market_cap_usd=1000.0, max_capital_usd=10_000.0,
    )
    engine, journal, _v, _q = _engine(tmp_path, venue=venue, config=cfg)
    engine.handle_signal(_signal(condition_id="0xc1"))
    engine.handle_signal(_signal(condition_id="0xc2"))
    assert engine.is_halted() is False
    engine.handle_signal(_signal(condition_id="0xc3"))
    assert engine.is_halted() is True
    assert len(venue.calls) == 2  # third never reached venue
    journal.close()
    halts = [e for e in journal.read_today() if e["event_type"] == "RISK_HALT"]
    reasons = [h["reason"] for h in halts]
    assert "max_real_orders" in reasons


def test_max_real_orders_does_not_fire_for_fake_venue(tmp_path: Path) -> None:
    venue = _FakeVenue(real=False)
    cfg = _config(
        max_real_orders=1, per_trade_cap_usd=1000.0,
        per_market_cap_usd=1000.0, max_capital_usd=10_000.0,
    )
    engine, journal, _v, _q = _engine(tmp_path, venue=venue, config=cfg)
    for i in range(5):
        engine.handle_signal(_signal(condition_id=f"0xc{i}"))
    assert engine.is_halted() is False
    journal.close()
    halts = [e for e in journal.read_today() if e["event_type"] == "RISK_HALT"]
    reasons = [h["reason"] for h in halts]
    # max_real_orders must NOT appear — it's gated on is_real_venue.
    assert "max_real_orders" not in reasons


def test_operator_aborted_skips_one_order_without_halting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    venue = _FakeVenue(real=True, immediate_fill=False)
    cfg = _config(
        require_operator_confirm=True, per_trade_cap_usd=1000.0,
        per_market_cap_usd=1000.0, max_capital_usd=10_000.0,
    )
    engine, journal, _v, _q = _engine(tmp_path, venue=venue, config=cfg)
    monkeypatch.setattr("sys.stdin", _StdinStub("no\n"))
    engine.handle_signal(_signal())
    assert engine.is_halted() is False
    assert venue.calls == []  # didn't reach the venue
    journal.close()
    halts = [e for e in journal.read_today() if e["event_type"] == "RISK_HALT"]
    assert any(h["reason"] == "operator_aborted" for h in halts)


def test_operator_yes_proceeds_with_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    venue = _FakeVenue(real=True, immediate_fill=False)
    cfg = _config(
        require_operator_confirm=True, per_trade_cap_usd=1000.0,
        per_market_cap_usd=1000.0, max_capital_usd=10_000.0,
    )
    engine, _journal, _v, _q = _engine(tmp_path, venue=venue, config=cfg)
    monkeypatch.setattr("sys.stdin", _StdinStub("yes\n"))
    engine.handle_signal(_signal())
    assert len(venue.calls) == 1


def test_order_submitted_journaled_before_venue_call(tmp_path: Path) -> None:
    """Even when the venue raises, OrderSubmitted is in the journal."""
    venue = _FakeVenue(raise_exc=ConnectionError("boom"))
    engine, journal, _v, _q = _engine(tmp_path, venue=venue)
    engine.handle_signal(_signal())
    journal.close()
    events = list(journal.read_today())
    types_in_order = [e["event_type"] for e in events]
    submitted_index = types_in_order.index("ORDER_SUBMITTED")
    rejected_index = types_in_order.index("ORDER_REJECTED")
    assert submitted_index < rejected_index


class _StdinStub:
    """Minimal stdin stand-in for sys.stdin.readline()."""
    def __init__(self, line: str) -> None:
        self._line = line

    def readline(self) -> str:
        return self._line
