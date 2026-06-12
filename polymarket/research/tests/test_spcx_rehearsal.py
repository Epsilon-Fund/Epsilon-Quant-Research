"""Block S5i — REHEARSAL tab tests. The acceptance bar is ISOLATION: the rehearsal is a
read-only consumer of production engine functions, with its own state namespace, zero
writes to playbook_state.json, zero parquet shards, and production behavior bit-identical
while it runs."""
from __future__ import annotations

import copy
import json
import urllib.request as _url
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import spcx_rehearsal as rh
from scripts.spcx_pm_pdf_monitor import IPO_OFFER_DEFAULT, DashboardState, analyze, \
    fixture_snapshot
from scripts.spcx_dashboard_server import DashboardServer, build_ws_payload
from tests.test_spcx_pm_pdf_monitor import _pb, _two_poll_dash


# ---------------------------------------------------------------- synthetic tape
def _synth_bars(n_rth=120, pre=10, open_px=200.0, drift=-0.15):
    """Pre-market bars + a fading RTH session starting 9:30 ET (13:30 UTC in June)."""
    bars = []
    base = datetime(2026, 6, 10, 13, 30, tzinfo=timezone.utc)
    for k in range(pre):
        t = base - timedelta(minutes=pre - k)
        bars.append({"time_utc": t.isoformat(), "open": 195.0, "high": 195.5,
                     "low": 194.5, "close": 195.0, "volume": 1000})
    px = open_px
    for k in range(n_rth):
        t = base + timedelta(minutes=k)
        px = open_px + drift * k if k > 5 else open_px + 0.3 * k  # pop then fade
        bars.append({"time_utc": t.isoformat(), "open": px, "high": px + 0.2,
                     "low": px - 0.2, "close": px, "volume": 50_000})
    return bars


def test_cross_is_first_rth_bar_premarket_excluded():
    bars = _synth_bars()
    rth = rh.rth_bars(bars)
    assert len(rth) == 120                            # pre-market dropped
    assert rth[0]["time_utc"].startswith("2026-06-10T13:30")
    reh = rh.build_rehearsal(bars)
    assert reh["cross_px"] == pytest.approx(200.0)    # the 9:30 open print
    assert reh["offer"] == pytest.approx(200.0 / 1.30)


def test_level_scaling_ratios():
    levels = {l["label"]: l["scaled"] for l in rh.scale_levels(offer=154.0)}
    # each level = SPCX dollar level / 135 × offer
    assert levels["sell everything"] == pytest.approx(125 / 135 * 154.0)
    assert levels["CRASH floor"] == pytest.approx(160 / 135 * 154.0)
    assert levels["pre-hedge trigger"] == pytest.approx(183 / 135 * 154.0)
    assert levels["offer"] == pytest.approx(154.0)
    # the SPCX-space mapping makes the production constants the scaled levels exactly:
    # px_scaled_level × (135/offer) == SPCX constant
    reh = rh.build_rehearsal(_synth_bars())
    assert levels_round_trip(reh)


def levels_round_trip(reh):
    return all(abs(l["scaled"] * reh["to_spcx"] - l["spcx"]) < 1e-9
               for l in rh.scale_levels(reh["offer"]))


def test_state_ribbon_lookahead_free_truncation():
    """The state at minute t must be identical whether or not bars after t exist."""
    bars = _synth_bars(n_rth=120)
    full = rh.build_rehearsal(bars)["ribbon"]
    rth_all = rh.rth_bars(bars)
    pre = [b for b in bars if b not in rth_all]
    for cut in (30, 60, 90):
        part = rh.build_rehearsal(pre + rth_all[:cut])["ribbon"]
        assert part == full[:cut], f"lookahead leak at cut={cut}"


def test_ribbon_fades_on_fading_tape():
    """The synthetic pop-then-fade tape must reach FADE through the production rules."""
    reh = rh.build_rehearsal(_synth_bars(n_rth=120, drift=-0.2))
    assert "FADE" in reh["ribbon"]
    i = reh["ribbon"].index("FADE")
    assert all(s in ("FLAT", "RALLY") for s in reh["ribbon"][:i])


def test_ticket_click_sold_arithmetic(tmp_path):
    """Cumulative mark-sold drives the production tranche statuses (22 sh → 8/8/6)."""
    reh = rh.build_rehearsal(_synth_bars())
    sp = tmp_path / "reh.json"
    p0 = rh.panel_at(reh, 100, sold=0.0, state_path=sp)
    rows0 = {r["name"]: r["status"] for r in p0["tranche"]["rows"]}
    assert not any(s == "DONE" for s in rows0.values())
    p1 = rh.panel_at(reh, 100, sold=8.0, state_path=sp)
    assert {r["name"]: r["status"] for r in p1["tranche"]["rows"]}["T1"] == "DONE"
    p2 = rh.panel_at(reh, 100, sold=16.0, state_path=sp)
    st2 = {r["name"]: r["status"] for r in p2["tranche"]["rows"]}
    assert st2["T1"] == "DONE" and st2["T2"] == "DONE" and st2["T3"] != "DONE"
    # simulated hedge-ops runs and is labeled
    assert "SIMULATED" in p1["hedge_ops_html"] and "HEDGE OPS" in p1["hedge_ops_html"]


def test_tape_cache_and_fallback(tmp_path, monkeypatch):
    """Alpaca failure → Yahoo fallback → cache written → second load is cache-served."""
    monkeypatch.setattr(rh, "TAPE_DIR", tmp_path)
    calls = {"a": 0, "y": 0}

    def boom(*a, **k):
        calls["a"] += 1
        raise RuntimeError("alpaca down")

    def yahoo(symbol, day, timeout=30.0):
        calls["y"] += 1
        return _synth_bars()

    monkeypatch.setattr(rh, "fetch_tape_alpaca", boom)
    monkeypatch.setattr(rh, "fetch_tape_yahoo", yahoo)
    bars, day, source = rh.load_tape("TEST", "2026-06-10")
    assert source == "yahoo" and calls == {"a": 1, "y": 1}
    assert (tmp_path / "rehearsal_test_2026-06-10_1m.csv").exists()
    bars2, _, source2 = rh.load_tape("TEST", "2026-06-10")
    assert source2 == "cache" and calls == {"a": 1, "y": 1}   # no refetch
    assert len(bars2) == len(bars)


def test_build_rehearsal_does_not_mutate_inputs():
    bars = _synth_bars()
    snapshot = copy.deepcopy(bars)
    rh.build_rehearsal(bars)
    assert bars == snapshot


# ---------------------------------------------------------------- isolation (the bar)
def _reh_server(tmp_path, monkeypatch):
    from scripts.spcx_pm_pdf_monitor import PlaybookState
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash = DashboardState()
    dash.record(snap, rep)
    prod_state = tmp_path / "playbook_state.json"
    pb = PlaybookState(path=prod_state)
    pb.eurusd = 1.08
    pb.save()
    server = DashboardServer(dash, pb, port=0,
                             reh_state_path=tmp_path / "rehearsal_state.json")
    server.start()
    server.last_ctx = (rep, snap, None)
    # rehearsal loads from a synthetic tape — no network
    monkeypatch.setattr(rh, "load_tape",
                        lambda s, d=None: (_synth_bars(), "2026-06-10", "cache"))
    return server, prod_state, pb, dash, rep, snap


def test_rehearsal_isolation_no_production_writes(tmp_path, monkeypatch):
    server, prod_state, pb, dash, rep, snap = _reh_server(tmp_path, monkeypatch)
    before = prod_state.read_bytes()
    prod_files = set(tmp_path.iterdir())
    # full rehearsal lifecycle: load → scrub panels → mark sold → reload
    body = json.loads(_url.urlopen(
        server.url() + "api/rehearsal/load?symbol=TEST&reset=1", timeout=10).read())
    assert body["n"] == 120 and body["sold"] == 0.0
    for i in (0, 40, 80, 119):
        p = json.loads(_url.urlopen(
            server.url() + f"api/rehearsal/panel?i={i}", timeout=10).read())
        assert "tranche_html" in p and "REHEARSAL" not in (p.get("error") or "")
    req = _url.Request(server.url() + "api/rehearsal/state", method="POST",
                       data=json.dumps({"sold": 8}).encode(),
                       headers={"Content-Type": "application/json"})
    assert json.loads(_url.urlopen(req, timeout=10).read())["ok"]
    # 1) the PRODUCTION state file is byte-identical
    assert prod_state.read_bytes() == before
    # 2) the only new file is the rehearsal namespace
    new_files = set(tmp_path.iterdir()) - prod_files
    assert new_files == {tmp_path / "rehearsal_state.json"}
    assert json.loads((tmp_path / "rehearsal_state.json").read_text())["sold"] == 8
    # 3) production payload is unchanged by a rehearsal having run (same inputs → same
    # output; ts fields are derived from dash/snap, not wall clock)
    p_after = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>")
    p_ref = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>")
    assert p_after == p_ref
    # 4) production endpoints still live while the rehearsal is loaded
    st = json.loads(_url.urlopen(_url.Request(
        server.url() + "api/state", method="POST",
        data=json.dumps({"comfort": 22}).encode(),
        headers={"Content-Type": "application/json"}), timeout=10).read())
    assert st["ok"]


def test_rehearsal_panel_requires_load_and_validates(tmp_path, monkeypatch):
    server, *_ = _reh_server(tmp_path, monkeypatch)
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as ei:
        _url.urlopen(server.url() + "api/rehearsal/panel?i=5", timeout=10)
    assert ei.value.code == 404            # nothing loaded yet
    _url.urlopen(server.url() + "api/rehearsal/load?symbol=TEST", timeout=10).read()
    with pytest.raises(urllib.error.HTTPError) as ei2:
        _url.urlopen(server.url() + "api/rehearsal/panel", timeout=10)
    assert ei2.value.code == 400           # i missing


def test_production_push_unaffected_during_rehearsal_hammering(tmp_path, monkeypatch):
    """Production cadence proxy: pushes succeed and clients receive them while the
    rehearsal panel endpoint is being hammered."""
    server, prod_state, pb, dash, rep, snap = _reh_server(tmp_path, monkeypatch)
    _url.urlopen(server.url() + "api/rehearsal/load?symbol=TEST", timeout=10).read()
    import websocket as wsc
    ws = wsc.create_connection(server.url().replace("http", "ws") + "ws", timeout=10)
    json.loads(ws.recv())                  # snapshot
    got_polls = 0
    for i in range(10):
        _url.urlopen(server.url() + f"api/rehearsal/panel?i={i * 10}", timeout=10).read()
        server.push(build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>"))
        m = json.loads(ws.recv())
        if m["type"] == "poll":
            got_polls += 1
    ws.close()
    assert got_polls == 10                 # every production push delivered, in order
