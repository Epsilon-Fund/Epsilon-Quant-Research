"""CLI-level dry-run E2E of ``--mode mm_bridge`` (GOAL-2 acceptance) — no network.

Runs the REAL operator entry point (``mm_bridge_cli.main``) end-to-end with: env-driven
config (1-contract, operator-confirm on, reconcile-every set — the Join-2c wiring), the
fake venue, a monkeypatched market-data feed (three book frames), and a monkeypatched
Data-API client (no HTTP). Asserts a clean exit, the session lifecycle journal events, and
that the startup banner surfaces every safety-relevant knob.
"""
from __future__ import annotations

import json
from pathlib import Path

from polymarket.execution.maker import mm_bridge_cli
from polymarket.execution.maker.mm_engine_bridge import ensure_mm_engine_importable

ensure_mm_engine_importable()
from mm_engine.interfaces import MarketEvent  # noqa: E402


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "POLYMARKET_LEADER_ADDRESS": "0x" + "1" * 40,
        "POLYMARKET_PRIVATE_KEY": "key",
        "POLYMARKET_API_KEY": "api",
        "POLYMARKET_API_SECRET": "secret",
        "POLYMARKET_PASSPHRASE": "pass",
        "POLYMARKET_FUNDER": "0x" + "2" * 40,
        "POLYMARKET_JOURNAL_DIR": str(tmp_path / "journal"),
        "POLYMARKET_KILLSWITCH_PATH": str(tmp_path / "no_killswitch"),
        # --- the Join-2c wiring under test ---
        "POLYMARKET_VENUE": "fake",                       # DRY-RUN
        "POLYMARKET_MAKER_CONDITION_ID": "0xcond",
        "POLYMARKET_MM_BRIDGE_ASSET_ID": "t1",            # skip the Gamma lookup (no network)
        "MAKER_SIZE_CONTRACTS": "1",                      # 1-contract caps
        "POLYMARKET_REQUIRE_OPERATOR_CONFIRM": "true",
        "POLYMARKET_MAX_REAL_ORDERS": "1",
        "POLYMARKET_MM_BRIDGE_RECONCILE_EVERY": "100",    # rate-limit-safe venue reconcile
        "POLYMARKET_MM_BRIDGE_LATENCY_MS": "203",         # the 2b fit landing in the model
        "POLYMARKET_MM_BRIDGE_MAX_INVENTORY": "5",        # hard contracts cap
        "POLYMARKET_MM_BRIDGE_MAX_EVENTS": "3",
    }


def _fake_feed(asset_ids, max_events=None, **_kwargs):
    for i, ts in enumerate((1000, 1100, 1200)):
        if max_events is not None and i >= max_events:
            return
        yield MarketEvent(
            type="book", token_id=asset_ids[0], ts_exchange=ts, ts_local_iso="",
            ts_monotonic_ns=0,
            payload={"asset_id": asset_ids[0],
                     "bids": [{"price": 0.47, "size": 100}],
                     "asks": [{"price": 0.49, "size": 100}]},
        )


class _NoNetworkDataClient:
    def __init__(self, *_a, **_k) -> None:
        pass

    def get_trades(self, condition_id: str) -> list[dict]:  # noqa: ARG002
        return []


def test_mm_bridge_cli_dry_run_end_to_end(tmp_path: Path, monkeypatch, capsys) -> None:
    import mm_engine.feeds.live_shadow as live_shadow

    monkeypatch.setattr(live_shadow, "live_shadow_feed", _fake_feed)
    monkeypatch.setattr(mm_bridge_cli, "DataApiTradeClient", _NoNetworkDataClient)

    rc = mm_bridge_cli.main(_env(tmp_path))
    assert rc == 0

    out = capsys.readouterr().out
    # the startup banner surfaces every safety-relevant knob of the 2c wiring
    assert "venue=fake" in out
    assert "max_real_orders=1" in out
    assert "operator_confirm=True" in out
    assert "latency_ms=203.0" in out
    assert "max_inventory=5.0" in out
    assert "reconcile_every=100" in out
    assert "WARNING" not in out          # both live-run knobs set → no startup warnings
    assert "clean exit" in out

    # session lifecycle journaled
    journal_files = list((tmp_path / "journal").glob("mm_bridge-*.jsonl"))
    assert journal_files, "mm_bridge journal file missing"
    events = [json.loads(line) for line in journal_files[0].read_text().splitlines()]
    types = [e.get("event_type") for e in events]
    assert "MAKER_SESSION_STARTED" in types
    assert "MAKER_SESSION_STOPPED" in types
    # DRY-RUN quotes flowed (two-sided on the fake venue → placements journaled)
    assert "MAKER_QUOTE_PLACED" in types


def test_mm_bridge_cli_warns_when_live_knobs_unset(tmp_path: Path, monkeypatch, capsys) -> None:
    import mm_engine.feeds.live_shadow as live_shadow

    monkeypatch.setattr(live_shadow, "live_shadow_feed", _fake_feed)
    monkeypatch.setattr(mm_bridge_cli, "DataApiTradeClient", _NoNetworkDataClient)

    env = _env(tmp_path)
    env.pop("POLYMARKET_MM_BRIDGE_RECONCILE_EVERY")
    env.pop("POLYMARKET_MM_BRIDGE_MAX_INVENTORY")
    rc = mm_bridge_cli.main(env)
    assert rc == 0

    out = capsys.readouterr().out
    assert "POLYMARKET_MM_BRIDGE_MAX_INVENTORY is unset/0" in out
    assert "POLYMARKET_MM_BRIDGE_RECONCILE_EVERY is unset/0" in out
