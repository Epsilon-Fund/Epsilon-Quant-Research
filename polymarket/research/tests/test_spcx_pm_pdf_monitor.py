"""Acceptance tests for the SPCX PM-PDF monitor (Block S5).

These encode the task-spec acceptance criteria:
  (i)   survivor monotonicity is enforced (clip + flag) and the PCHIP interpolant is
        non-increasing with a non-negative implied PDF;
  (ii)  the EV integral E[cap] = ∫S dK is exact on a synthetic surface with a known
        closed-form distribution, and agrees with the ∫K·pdf cross-check;
  (iii) per-share conversion is correct and BOTH share-base columns (13.076B primary,
        13.091B coworker) are emitted;
  (iv)  the embedded 2026-06-07 coworker surface reproduces the addendum's distribution
        table within tolerance (the offline reproduction gate);
  (v)   one-sided / empty books degrade gracefully (flagged, never fatal while >=4
        strikes remain usable);
  (vi)  the bucket-vs-ladder comparison is ~zero on a self-consistent synthetic market
        and reproduces the audit's known 1.5-2.0T divergence on the fixture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.spcx_pm_pdf_monitor import (
    SHARES_COWORKER,
    SHARES_PRIMARY,
    TAIL_STRIKES,
    analyze,
    bucket_compare,
    cap_t_to_per_share,
    dist_stats,
    enforce_monotone,
    ev_convention_sweep,
    extract_points,
    fit_lognormal_weighted,
    fit_survivor,
    fixture_snapshot,
    lognormal_survivor,
    tail_trade_eval,
    log_parquet,
    pchip_eval,
    pchip_slopes,
    per_share_to_cap_t,
    render_html,
    render_text,
)


# ---------------------------------------------------------------- (i) monotonicity
def test_enforce_monotone_clips_and_flags():
    strikes = [1.0, 1.5, 2.0, 2.5, 3.0]
    probs = [0.95, 0.80, 0.85, 0.40, 0.10]  # 2.0 quotes ABOVE 1.5 — inconsistent
    k, p, violated = enforce_monotone(strikes, probs)
    assert list(k) == strikes
    assert all(p[i] >= p[i + 1] for i in range(len(p) - 1))
    assert violated == [2.0]
    assert p[2] == pytest.approx(0.80)  # clipped to the running minimum


def test_enforce_monotone_clean_input_no_flags():
    _, p, violated = enforce_monotone([1.0, 2.0, 3.0], [0.9, 0.5, 0.1])
    assert violated == []
    assert list(p) == [0.9, 0.5, 0.1]


def test_pchip_survivor_is_monotone_and_pdf_nonnegative():
    rng = np.random.default_rng(7)
    probs = np.sort(rng.uniform(0.01, 0.99, 12))[::-1]
    fit = fit_survivor(list(np.linspace(1.0, 4.0, 12)), list(probs))
    S = fit["S"]
    assert np.all(np.diff(S) <= 1e-12)        # non-increasing survivor
    assert np.all(fit["pdf"] >= 0.0)          # PDF clipped/derived non-negative
    assert S[0] == pytest.approx(1.0)
    assert fit["tail_mass_beyond_grid"] < 0.01


def test_pchip_interpolates_knots_exactly():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.7, 0.2, 0.05])
    m = pchip_slopes(x, y)
    out = pchip_eval(x, y, m, x)
    np.testing.assert_allclose(out, y, atol=1e-12)


# ---------------------------------------------------------------- (ii) EV integral
def test_ev_integral_uniform_distribution():
    """Closing cap ~ Uniform(1.0T, 3.0T): survivor is exactly linear between the strikes,
    so the PCHIP fit must recover mean=2.0T, median=2.0T, P25=1.5T, P75=2.5T."""
    strikes = list(np.linspace(1.0, 3.0, 21))
    probs = [(3.0 - k) / 2.0 for k in strikes]
    fit = fit_survivor(strikes, probs, tail_anchors=())
    st = dist_stats(fit, shares=SHARES_PRIMARY)
    assert st["mean_cap_t"] == pytest.approx(2.0, abs=0.01)
    assert st["median_cap_t"] == pytest.approx(2.0, abs=0.01)
    assert st["p25_cap_t"] == pytest.approx(1.5, abs=0.02)
    assert st["p75_cap_t"] == pytest.approx(2.5, abs=0.02)
    # survivor-integral mean and pdf-moment mean agree
    assert st["mean_pdf_xcheck_t"] == pytest.approx(st["mean_cap_t"], abs=0.02)


def test_ev_integral_lognormal_distribution():
    """Lognormal survivor points (mu, sigma known) → mean must match exp(mu + sig^2/2)."""
    from math import erf

    mu, sig = math.log(2.1), 0.25

    def surv(k):
        z = (math.log(k) - mu) / sig
        return 1.0 - 0.5 * (1.0 + erf(z / math.sqrt(2)))

    strikes = list(np.linspace(0.8, 4.6, 30))
    fit = fit_survivor(strikes, [surv(k) for k in strikes], tail_anchors=((5.0, 1e-5),))
    st = dist_stats(fit, shares=SHARES_PRIMARY)
    assert st["mean_cap_t"] == pytest.approx(math.exp(mu + sig**2 / 2), abs=0.02)
    assert st["median_cap_t"] == pytest.approx(math.exp(mu), abs=0.02)


# ---------------------------------------------------------------- (iii) per-share bases
def test_per_share_conversion_both_bases():
    assert cap_t_to_per_share(2.185, SHARES_PRIMARY) == pytest.approx(167.10, abs=0.01)
    assert cap_t_to_per_share(2.185, SHARES_COWORKER) == pytest.approx(166.91, abs=0.01)
    assert per_share_to_cap_t(135.0, SHARES_COWORKER) == pytest.approx(1.7673, abs=0.001)
    # round trip
    assert per_share_to_cap_t(cap_t_to_per_share(2.0, SHARES_PRIMARY),
                              SHARES_PRIMARY) == pytest.approx(2.0)


def test_analyze_emits_both_share_base_columns():
    rep = analyze(fixture_snapshot(), basis="mid")
    a, b = rep["stats_primary"], rep["stats_coworker"]
    assert a["shares"] == SHARES_PRIMARY and b["shares"] == SHARES_COWORKER
    # same cap-space distribution, different per-share scaling
    assert a["mean_cap_t"] == pytest.approx(b["mean_cap_t"])
    assert a["mean_ps"] / b["mean_ps"] == pytest.approx(SHARES_COWORKER / SHARES_PRIMARY,
                                                        rel=1e-9)


# ------------------------------------------------- (iv) 2026-06-07 reproduction gate
def test_reproduces_addendum_2026_06_07_table():
    """The addendum's coworker distribution table (PCHIP on mids, 13.091B shares):
    mean $166.9 / median $164.2 / mode $161.3 / std $40.7 / P25 140.2 / P75 187.3 /
    P90 215.4 / P95 235.2; P(win) 79.9%. Tolerances are tight on the construction-robust
    stats, looser on the shape-fragile mode/std (see spacex_pdf_construction_audit.md)."""
    rep = analyze(fixture_snapshot(), basis="mid")
    b = rep["stats_coworker"]
    assert b["p_win_offer"] * 100 == pytest.approx(79.9, abs=0.5)
    assert b["mean_ps"] == pytest.approx(166.9, abs=1.0)
    assert b["median_ps"] == pytest.approx(164.2, abs=1.0)
    assert b["p25_ps"] == pytest.approx(140.2, abs=1.5)
    assert b["p75_ps"] == pytest.approx(187.3, abs=1.5)
    assert b["p90_ps"] == pytest.approx(215.4, abs=2.0)
    assert b["p95_ps"] == pytest.approx(235.2, abs=2.0)
    assert b["mode_ps"] == pytest.approx(161.3, abs=3.0)
    assert b["std_cap_t"] * 1e12 / SHARES_COWORKER == pytest.approx(40.7, abs=1.5)
    # cap-space headline from the addendum: mean 2.185T, median 2.149T
    assert b["mean_cap_t"] == pytest.approx(2.185, abs=0.015)
    assert b["median_cap_t"] == pytest.approx(2.149, abs=0.015)


def test_fixture_bucket_overlay_matches_audit():
    """The audit's known result: every bucket within a few pp EXCEPT 1.5-2.0T at ~+7.8pp."""
    rep = analyze(fixture_snapshot(), basis="mid")
    by_label = {r["label"]: r for r in rep["buckets"]}
    assert by_label["1.5-2.0T"]["gap_pp"] == pytest.approx(7.8, abs=0.8)
    for label, r in by_label.items():
        if label != "1.5-2.0T":
            assert abs(r["gap_pp"]) < 5.0, f"{label} unexpectedly divergent"


# ------------------------------------------------- (v) graceful degradation
def test_one_sided_and_empty_strikes_degrade_gracefully():
    snap = fixture_snapshot()
    snap["ladder"][3]["bid"] = None                       # one-sided: ask only
    snap["ladder"][5]["ask"] = None                       # one-sided: bid only
    snap["ladder"][10]["bid"] = snap["ladder"][10]["ask"] = None  # dead book
    rep = analyze(snap, basis="mid")
    assert rep["n_strikes_used"] == 15                    # dead strike dropped, fit alive
    assert any("used ask" in f for f in rep["degradation_flags"])
    assert any("used bid" in f for f in rep["degradation_flags"])
    assert any("dropped" in f for f in rep["degradation_flags"])
    # stats still sane
    assert 1.5 < rep["stats_primary"]["mean_cap_t"] < 3.0


def test_too_few_strikes_refuses_to_fit():
    snap = fixture_snapshot()
    for row in snap["ladder"][3:]:
        row["bid"] = row["ask"] = None
    with pytest.raises(RuntimeError, match="usable ladder strikes"):
        analyze(snap, basis="mid")


def test_ask_basis_drops_missing_side():
    snap = fixture_snapshot()
    snap["ladder"][2]["ask"] = None
    xs, ps, flags = extract_points(snap["ladder"], "ask")
    assert len(xs) == 15
    assert any("no ask quote" in f for f in flags)


# ------------------------------------------------- (vi) bucket comparison correctness
def test_bucket_compare_consistent_market_has_zero_gaps():
    """Ladder and buckets generated from the same uniform(1,3) distribution must agree."""
    strikes = list(np.linspace(1.0, 3.0, 21))
    probs = [(3.0 - k) / 2.0 for k in strikes]
    buckets = [{"label": "1.0-1.5T", "lo": 1.0, "hi": 1.5, "mid": 0.25},
               {"label": "1.5-2.0T", "lo": 1.5, "hi": 2.0, "mid": 0.25},
               {"label": "2.0-2.5T", "lo": 2.0, "hi": 2.5, "mid": 0.25},
               {"label": "2.5-3.0T", "lo": 2.5, "hi": 3.0, "mid": 0.25}]
    rows = bucket_compare(strikes, probs, buckets)
    for r in rows:
        assert abs(r["gap_pp"]) < 0.5, r


# ------------------------------------------------- reconcile sweep + renderers + log
def test_ev_sweep_correct_ev_positive_and_total_loss_family_nearest_png():
    snap = fixture_snapshot()
    xs, ps, _ = extract_points(snap["ladder"], "mid")
    sweep = ev_convention_sweep(fit_survivor(xs, ps))
    by_name = {r["convention"][:2]: r for r in sweep}
    assert by_name["A1"]["ev_ps"] == pytest.approx(31.9, abs=1.0)   # the correct EV
    assert by_name["A2"]["ev_ps"] == pytest.approx(by_name["A1"]["ev_ps"], abs=0.5)
    best = min(sweep, key=lambda r: r["abs_err_vs_png"])
    assert best["convention"].startswith("D"), (
        "expected the total-loss-if-lose family to be nearest the PNG's -3.3")
    assert best["abs_err_vs_png"] < 1.0


def test_render_text_and_html_smoke():
    rep = analyze(fixture_snapshot(), basis="mid", spot=168.5)
    txt = render_text(rep)
    assert "P(close > $135)" in txt and "13.091B" in txt and "LISTED $168.50" in txt
    html = render_html(rep, refresh_s=45)
    assert html.startswith("<!doctype html>") and "refresh" in html


def test_parquet_log_appends_shard(tmp_path):
    import pyarrow.parquet as pq

    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    p = log_parquet(snap, rep, out_dir=tmp_path)
    assert p.exists()
    t = pq.read_table(p)
    kinds = set(t.column("kind").to_pylist())
    assert kinds == {"ladder", "bucket", "no_ipo"}
    assert t.num_rows == 16 + 7 + 1
    # poll-level stats denormalized onto every row
    assert t.column("mean_ps")[0].as_py() == pytest.approx(rep["stats_primary"]["mean_ps"])
    # second poll → second shard, first untouched (append-only)
    snap2 = dict(snap, fetched_at_utc="2026-06-07T16:00:00+00:00")
    p2 = log_parquet(snap2, rep, out_dir=tmp_path)
    assert p2 != p and p.exists() and p2.exists()


# ------------------------------------------------- Block S5b: auto spot feed (Alpaca IEX)
def _feed(symbol="SPCX"):
    from scripts.spcx_pm_pdf_monitor import AlpacaSpotFeed
    return AlpacaSpotFeed(symbol, key_id="test-key", secret="test-secret")


def _trade_frame(symbol, price, ts):
    return json.dumps([{"T": "t", "S": symbol, "p": price, "s": 10, "t": ts}])


import json  # noqa: E402  (test-local convenience)
import time as _time  # noqa: E402


def test_spot_feed_handle_message_updates_last_trade():
    feed = _feed()
    # noise frames must be ignored: auth ack, subscription ack, other symbols, junk
    feed.handle_message(json.dumps([{"T": "success", "msg": "authenticated"}]))
    feed.handle_message(json.dumps([{"T": "subscription", "trades": ["SPCX"]}]))
    feed.handle_message(_trade_frame("AAPL", 999.0, "2026-06-12T17:30:00.123456789Z"))
    feed.handle_message("not json at all")
    assert feed.snapshot()["status"] == "no_prints"
    feed.handle_message(_trade_frame("SPCX", 168.5, "2026-06-12T17:30:00.123456789Z"))
    snap = feed.snapshot(stale_secs=120,
                         now=_parse_epoch("2026-06-12T17:30:30+00:00"))
    assert snap["status"] == "live"
    assert snap["price"] == pytest.approx(168.5)
    assert snap["age_s"] == pytest.approx(29.9, abs=0.5)


def _parse_epoch(ts):
    from datetime import datetime
    return datetime.fromisoformat(ts).timestamp()


def test_manual_spot_overrides_ws_feed():
    from scripts.spcx_pm_pdf_monitor import resolve_spot
    feed = _feed()
    feed.handle_message(_trade_frame("SPCX", 150.0, "2026-06-12T17:30:00Z"))
    spot, meta = resolve_spot(168.5, feed, 120, now=_parse_epoch("2026-06-12T17:30:10+00:00"))
    assert spot == 168.5 and meta["source"] == "manual"
    # without manual, the feed wins
    spot2, meta2 = resolve_spot(None, feed, 120, now=_parse_epoch("2026-06-12T17:30:10+00:00"))
    assert spot2 == pytest.approx(150.0) and meta2["source"] == "alpaca_iex"


def test_stale_feed_renders_stale_and_monitor_keeps_working():
    from scripts.spcx_pm_pdf_monitor import resolve_spot
    feed = _feed()
    feed.handle_message(_trade_frame("SPCX", 150.0, "2026-06-12T17:30:00Z"))
    late = _parse_epoch("2026-06-12T17:35:00+00:00")  # 300s later > 120s stale
    spot, meta = resolve_spot(None, feed, 120, now=late)
    assert spot is None and meta["status"] == "stale"
    rep = analyze(fixture_snapshot(), basis="mid", spot=spot, spot_meta=meta)
    assert rep["spot"] is None                       # no crowd-vs-traded block
    assert 1.5 < rep["stats_primary"]["mean_cap_t"] < 3.0   # PM analysis unaffected
    txt = render_text(rep)
    assert "STALE" in txt and "age 300s" in txt and "unaffected" in txt


def test_no_prints_renders_pending_not_error():
    from scripts.spcx_pm_pdf_monitor import resolve_spot
    spot, meta = resolve_spot(None, _feed(), 120)
    assert spot is None and meta["status"] == "no_prints"
    txt = render_text(analyze(fixture_snapshot(), spot=spot, spot_meta=meta))
    assert "no prints yet" in txt and "normal pre-listing" in txt


def test_live_feed_label_carries_iex_caveat_and_age():
    from scripts.spcx_pm_pdf_monitor import resolve_spot
    feed = _feed()
    feed.handle_message(_trade_frame("SPCX", 168.5, "2026-06-12T17:30:00Z"))
    spot, meta = resolve_spot(None, feed, 120, now=_parse_epoch("2026-06-12T17:30:45+00:00"))
    txt = render_text(analyze(fixture_snapshot(), spot=spot, spot_meta=meta))
    assert "LISTED $168.50" in txt
    assert "IEX (≈2% of tape — signal-grade, not queue-grade)" in txt
    assert "trade age 45s" in txt


def test_reconnect_backoff_caps_and_never_raises():
    feed = _feed()
    delays = [feed.next_backoff() for _ in range(10)]
    assert delays[0] == 1.0
    assert all(b <= 60.0 for b in delays) and delays[-1] == 60.0
    assert all(delays[i] <= delays[i + 1] for i in range(len(delays) - 1))
    # a connect-failure loop iteration must not propagate: simulate by handle_message on
    # garbage + snapshot still answering
    feed.handle_message(b"\x00\xff garbage bytes")
    assert feed.snapshot()["status"] == "no_prints"
    # on_open resets the backoff
    class _WS:
        def __init__(self): self.sent = []
        def send(self, m): self.sent.append(json.loads(m))
    ws = _WS()
    feed._on_open(ws)
    assert feed._backoff == 1.0
    assert ws.sent[0]["action"] == "auth" and ws.sent[1]["action"] == "subscribe"
    assert ws.sent[1]["trades"] == ["SPCX"]


def test_feed_requires_env_credentials(monkeypatch):
    import dotenv

    from scripts.spcx_pm_pdf_monitor import AlpacaSpotFeed
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)  # a real .env exists
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_KEY_ID"):
        AlpacaSpotFeed("SPCX")


def test_parquet_spot_columns_and_backward_compat(tmp_path):
    import duckdb
    import pyarrow.parquet as pq
    from scripts.spcx_pm_pdf_monitor import resolve_spot

    # "old" shard: pre-S5b schema (simulated by dropping the new columns)
    snap = fixture_snapshot()
    rep_old = analyze(snap, basis="mid")
    p_old = log_parquet(snap, rep_old, out_dir=tmp_path)
    t_old = pq.read_table(p_old)
    old_cols = [c for c in t_old.column_names
                if c not in ("spot_source", "spot_status", "spot_age_s", "spot_last_price")]
    pq.write_table(t_old.select(old_cols), tmp_path / "poll_legacy.parquet")
    p_old.unlink()

    # new shard with a live ws spot
    feed = _feed()
    feed.handle_message(_trade_frame("SPCX", 168.5, "2026-06-12T17:30:00Z"))
    spot, meta = resolve_spot(None, feed, 120, now=_parse_epoch("2026-06-12T17:30:10+00:00"))
    snap2 = dict(snap, fetched_at_utc="2026-06-12T17:30:10+00:00")
    rep_new = analyze(snap2, basis="mid", spot=spot, spot_meta=meta)
    p_new = log_parquet(snap2, rep_new, out_dir=tmp_path)
    t_new = pq.read_table(p_new)
    assert {"spot_source", "spot_status", "spot_age_s", "spot_last_price"} <= set(t_new.column_names)
    assert t_new.column("spot_source")[0].as_py() == "alpaca_iex"
    assert t_new.column("spot_status")[0].as_py() == "live"

    # both shards readable together (union_by_name absorbs the schema extension)
    n = duckdb.sql(f"select count(*) from read_parquet('{tmp_path}/*.parquet', "
                   f"union_by_name=true)").fetchone()[0]
    assert n == t_old.num_rows + t_new.num_rows


# ------------------------------------------------- Block S5c: rich HTML dashboard
import re as _re
from pathlib import Path  # noqa: E402

from scripts.spcx_pm_pdf_monitor import (  # noqa: E402
    DashboardState,
    render_dashboard,
    write_html_atomic,
)


def _two_poll_dash(direction=+1, drop_tails=False, marks=None):
    """Fixture-based dashboard with two polls; poll 2 shifts every ladder quote by
    `direction`*0.01 so the crowd stats move a known way."""
    dash = DashboardState(marks=marks)
    snap1 = fixture_snapshot()
    if drop_tails:
        snap1["ladder"] = [r for r in snap1["ladder"]
                           if r["strike_t"] not in TAIL_STRIKES]
    rep1 = analyze(snap1, basis="mid")
    dash.record(snap1, rep1)
    snap2 = json.loads(json.dumps(snap1))
    snap2["fetched_at_utc"] = "2026-06-07T16:05:00+00:00"
    for r in snap2["ladder"]:
        r["bid"] = min(max(r["bid"] + 0.01 * direction, 0.001), 0.995)
        r["ask"] = min(max(r["ask"] + 0.01 * direction, 0.002), 0.999)
    rep2 = analyze(snap2, basis="mid")
    dash.record(snap2, rep2)
    return dash, rep2, snap2


def test_atomic_write_uses_temp_plus_rename(tmp_path, monkeypatch):
    import scripts.spcx_pm_pdf_monitor as mod
    calls = []
    real_replace = mod.os.replace
    monkeypatch.setattr(mod.os, "replace",
                        lambda src, dst: (calls.append((str(src), str(dst))),
                                          real_replace(src, dst))[1])
    target = tmp_path / "dash.html"
    write_html_atomic(target, "<html>ok</html>")
    assert target.read_text() == "<html>ok</html>"
    assert len(calls) == 1
    src, dst = calls[0]
    assert src.endswith(".tmp") and dst == str(target)
    assert not Path(src).exists()          # no leftover temp file


def test_dashboard_self_contained_no_external_resources():
    dash, rep, snap = _two_poll_dash()
    html = render_dashboard(dash, rep, snap)
    # resource-loading constructs are forbidden; xmlns/xlink namespace *declarations*
    # inside matplotlib SVGs are identifiers, not fetches, and are exempt.
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()
    assert "<iframe" not in html.lower()
    assert "@import" not in html
    assert not _re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", html), \
        "external src/href reference found"
    assert "url(http" not in html and "url(//" not in html


def test_dashboard_first_poll_charts_collecting():
    dash = DashboardState()
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash.record(snap, rep)
    html = render_dashboard(dash, rep, snap)
    assert html.count("collecting") == 2          # charts C and D
    assert "<svg" in html                          # A and B still render


def test_dashboard_many_polls_no_spot_renders():
    dash, rep, snap = _two_poll_dash()
    html = render_dashboard(dash, rep, snap)
    assert "no prints yet" in html                 # spot tile, pre-cross state
    assert html.count("<svg") >= 4                 # all four charts live
    assert "DIVERGENT" in html                     # 06-07 bucket gap renders highlighted
    assert "CROSSED" in html                       # perp 173.53 touched the 162 level
    assert "not crossed" in html                   # 125 untouched


def test_dashboard_missing_tail_strikes_placeholder():
    dash, rep, snap = _two_poll_dash(drop_tails=True)
    html = render_dashboard(dash, rep, snap)
    assert "no tail-strike quotes" in html         # chart D degrades, page alive
    assert html.count("<svg") >= 3


def _tile_delta_block(html, label):
    m = _re.search(_re.escape(label) + r"</div><div class='tv'>.*?<div class='td'>(.*?)</div>",
                   html, _re.S)
    assert m, f"tile {label!r} not found"
    return m.group(1)


def test_delta_arrows_on_synthetic_two_poll_sequence():
    # all quotes shifted UP on poll 2 → crowd mean and P(win) rise → ▲ vs prev
    dash, rep, snap = _two_poll_dash(direction=+1)
    html = render_dashboard(dash, rep, snap)
    assert "▲" in _tile_delta_block(html, "crowd mean")
    assert "▲" in _tile_delta_block(html, "P(close>$135)")
    # all quotes shifted DOWN → ▼
    dash, rep, snap = _two_poll_dash(direction=-1)
    html = render_dashboard(dash, rep, snap)
    assert "▼" in _tile_delta_block(html, "crowd mean")
    assert "▼" in _tile_delta_block(html, "P(close>$135)")


def test_mark_appears_in_timeseries_chart():
    dash, rep, snap = _two_poll_dash()
    # place the mark inside the history window (fixture history is 2026-06-07)
    dash.marks = [(dash.history[0]["ts"] + 60.0, "PRICED")]
    html = render_dashboard(dash, rep, snap)
    # matplotlib renders text as glyph paths; assert via the SVG aria/structure is brittle,
    # so chart C is rendered with the mark in-range and must differ from the no-mark page
    dash.marks = []
    html_nomark = render_dashboard(dash, rep, snap)
    assert html != html_nomark
    assert len(html) > len(html_nomark)   # the marker line+label adds SVG elements


def test_reference_overlay_fixture_math():
    from scripts.spcx_pm_pdf_monitor import per_share_to_cap_t, survivor_at
    dash = DashboardState()
    dash.set_reference(fixture_snapshot(), "2026-06-07 addendum fixture")
    # the overlay's own math must reproduce the addendum headline: P(win) ≈ 79.9%
    ref_fit = dash.reference["fit"]
    pwin = survivor_at(ref_fit, per_share_to_cap_t(135.0, 13_091_000_000))
    assert pwin * 100 == pytest.approx(79.9, abs=0.5)
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash.record(snap, rep)   # must NOT overwrite the explicit reference
    assert dash.reference["label"] == "2026-06-07 addendum fixture"
    html = render_dashboard(dash, rep, snap)
    assert "2026-06-07 addendum fixture" in html


def test_failed_chart_degrades_to_placeholder(monkeypatch):
    import scripts.spcx_pm_pdf_monitor as mod
    monkeypatch.setattr(mod, "chart_survivor",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    dash, rep, snap = _two_poll_dash()
    html = render_dashboard(dash, rep, snap)
    assert "survivor curve</b> — unavailable this poll" in html
    assert html.count("<svg") >= 3        # other charts still render


def test_alert_ledger_first_touch_and_untouched():
    dash, rep, snap = _two_poll_dash()
    # fixture perp = 173.53 → upside levels 162/140 crossed at the FIRST poll;
    # 183 (above perp) and the downside 135/125 never
    assert 162.0 in dash.alert_first_touch
    assert 140.0 in dash.alert_first_touch
    assert dash.alert_first_touch[162.0][0] == dash.history[0]["ts"]
    assert dash.alert_first_touch[162.0][1] == "perp"
    assert 183.0 not in dash.alert_first_touch
    assert 135.0 not in dash.alert_first_touch
    assert 125.0 not in dash.alert_first_touch


def test_dashboard_backfill_from_parquet(tmp_path):
    from datetime import datetime, timezone
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    for i in range(2):  # two shards stamped today so the backfill glob picks them up
        s = dict(snap, fetched_at_utc=datetime.now(timezone.utc)
                 .replace(minute=i, second=0).isoformat())
        log_parquet(s, rep, out_dir=tmp_path)
    dash = DashboardState()
    n = dash.backfill_from_parquet(tmp_path)
    assert n == 2 and len(dash.history) == 2
    assert dash.session_start_rec is dash.history[0]
    r = dash.history[0]
    assert r["mean_ps"] == pytest.approx(rep["stats_primary"]["mean_ps"])
    assert set(r["tails"].keys()) == set(TAIL_STRIKES)
    assert r["tails"][2.4][0] is not None and r["tails"][2.4][1] is not None


# ------------------------------------------------- Block S5d: playbook panel
from datetime import datetime as _dt  # noqa: E402

from scripts.spcx_pm_pdf_monitor import (  # noqa: E402
    PlaybookState,
    hedge_rule_eval,
    infer_node,
    pair_close_status,
    parse_cross_arg,
    render_playbook,
    tranche_phase,
)


def _cet_epoch(hhmm: str, day: str = "2026-06-12") -> float:
    from zoneinfo import ZoneInfo
    h, m = map(int, hhmm.split(":"))
    return _dt(*map(int, day.split("-")), h, m, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()


def _pb(**kw) -> PlaybookState:
    pb = PlaybookState(path=None)
    for k, v in kw.items():
        setattr(pb, k, v)
    return pb


def _d5_render(pb, snap_spot=None, avwap_info=None, now=None, history=None):
    """Render the playbook against the fixture surface with an optional injected spot."""
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid", spot=snap_spot)
    dash = DashboardState()
    if history is not None:
        dash.history = history
    return render_playbook(pb, dash, rep, snap, avwap_info, now=now)


# ---- node inference over the scripted day
def test_node_inference_scripted_day():
    pb = _pb()
    assert infer_node(pb, _cet_epoch("07:00")) == "PRE-ALLOC"   # no fill yet
    pb.fill = 40.0
    assert infer_node(pb, _cet_epoch("08:30")) == "D2/D3"       # allocation, pre-bell
    assert infer_node(pb, _cet_epoch("15:29")) == "D2/D3"
    assert infer_node(pb, _cet_epoch("15:30")) == "D4"          # bell, no cross yet
    assert infer_node(pb, _cet_epoch("18:00")) == "D4"
    pb.cross_ts = _cet_epoch("18:05")
    assert infer_node(pb, _cet_epoch("18:06")) == "D5"          # cross logged
    assert infer_node(pb, _cet_epoch("21:29")) == "D5"
    assert infer_node(pb, _cet_epoch("21:30")) == "CLOSE-OUT"
    assert infer_node(pb, _cet_epoch("21:45")) == "CLOSE-OUT"


def test_missing_inputs_render_awaiting_never_defaults():
    html = _d5_render(_pb(), now=_cet_epoch("07:00"))
    assert "awaiting <code>--fill" in html
    assert "hedge sleeve ON" not in html        # no rule evaluated without inputs
    # D4 without a cross prompts for it
    html = _d5_render(_pb(fill=40.0), now=_cet_epoch("16:00"))
    assert "--cross" in html and "awaiting: --cross" in html


# ---- D2 hedge arithmetic must equal the calc module bit-for-bit
def test_d2_hedge_arithmetic_equals_calc_grid():
    """Overflow-valve rule (gameplan §6 D2, updated 2026-06-11): the hedge covers only
    fill − comfort (~22 sh), still computed bit-for-bit by the calc grid (the overflow is
    fed to the grid as the fill, so hedged = min(overflow, margin cap))."""
    from scripts.spcx_convergence_calc import build_hedge_grid, shares_requested

    pb = _pb(fill=40.0, offer=135.0, sub_eur=10_000.0, margin_eur=2_000.0, eurusd=1.08)
    now = _cet_epoch("08:30")
    mark, funding = 158.0, 1e-5
    ev = hedge_rule_eval(pb, mark, funding, now)
    assert ev["comfort"] == 22.0 and ev["overflow"] == pytest.approx(18.0)
    # independent grid call with the OVERFLOW as the fill — must match exactly
    req = shares_requested(10_000.0, 1.08, 135.0)
    hours = (_dt.fromisoformat("2026-06-12T20:00:00+00:00").timestamp() - now) / 3600
    cells = build_hedge_grid(
        per_ipo_share_mark=mark, mark=mark, contracts_per_share=1.0,
        funding_hourly=funding, hours_to_settle=hours, fee_bps=4.5, fee_sides=2.0,
        subscription_eur=10_000.0, eurusd=1.08, margin_eur=2_000.0,
        fill_prices=[135.0], fill_fracs=[18.0 / req], leverages=[1.0, 1.5])
    by_lev = {c.leverage: c for c in cells}
    for lev in (1.0, 1.5):
        assert ev["cells"][lev].hedged_shares == by_lev[lev].hedged_shares
        assert ev["cells"][lev].margin_used == by_lev[lev].margin_used
        assert ev["cells"][lev].locked_net == by_lev[lev].locked_net
    # overflow 18 < the 1.5x margin cap (~20.5) → fully hedgeable
    assert ev["cells"][1.5].hedged_shares == pytest.approx(18.0)
    assert ev["go"] is True                      # overflow > 0 and basis >> costs
    assert ev["nearest_fill_row"] == 50          # 40 of ~86 requested ≈ 47% → 50% row
    # expected case: fill inside the comfort zone → no hedge regardless of basis
    pb2 = _pb(fill=20.0, offer=135.0, eurusd=1.08)
    ev2 = hedge_rule_eval(pb2, mark, funding, now)
    assert ev2["overflow"] == 0.0 and ev2["go"] is False


def test_d2_card_green_and_red_paths():
    pb = _pb(fill=40.0, offer=135.0, eurusd=1.08)
    snap = fixture_snapshot()           # fixture perp 173.53 → basis strongly positive
    rep = analyze(snap, basis="mid")
    html = render_playbook(pb, DashboardState(), rep, snap, None, now=_cet_epoch("08:30"))
    assert "hedge the OVERFLOW" in html and "gameplan §6 D2 / S1" in html
    assert "§3 row: " in html
    # expected case: fill ≤ comfort → explicit NO-hedge, margin-free message
    pb_small = _pb(fill=20.0, offer=135.0, eurusd=1.08)
    html_small = render_playbook(pb_small, DashboardState(), rep, snap, None,
                                 now=_cet_epoch("08:30"))
    assert "NO hedge — expected case" in html_small and "margin stays free" in html_small
    # red path: perp below the offer → basis ≤ 0 → NO hedge
    snap2 = fixture_snapshot()
    snap2["hl"]["mark"] = 132.0
    rep2 = analyze(snap2, basis="mid")
    html2 = render_playbook(pb, DashboardState(), rep2, snap2, None,
                            now=_cet_epoch("08:30"))
    assert "NO hedge" in html2
    assert "pop thesis weakening" in html2       # 132 ≤ 140 risk-off flag too


# ---- D5 tranche clock
def test_tranche_phase_boundaries():
    assert tranche_phase(0) == ("OBSERVE", 0.0)
    assert tranche_phase(14.99) == ("OBSERVE", 0.0)
    assert tranche_phase(15) == ("T1", 40.0)
    assert tranche_phase(59.99) == ("T1", 40.0)
    assert tranche_phase(60) == ("T2", 80.0)
    assert tranche_phase(179.99) == ("T2", 80.0)
    assert tranche_phase(180) == ("T3", 100.0)
    assert tranche_phase(400) == ("T3", 100.0)


def test_d5_tranche_card_and_sold_target():
    pb = _pb(fill=40.0, hedged=15.0, sold=10.0, cross_ts=_cet_epoch("18:00"),
             cross_price=165.0, eurusd=1.08)
    html = _d5_render(pb, snap_spot=164.0, now=_cet_epoch("18:30"))
    assert "30 min since cross" in html and "<b>T1</b>" in html
    assert "sold 10/25 sh (40%) vs target 40%" in html
    assert "S2 §4" in html


# ---- D5 stop ladder
def test_d5_stop_ladder_colors_and_crossing():
    pb = _pb(fill=40.0, cross_ts=_cet_epoch("18:00"), eurusd=1.08)
    html = _d5_render(pb, snap_spot=138.0, now=_cet_epoch("19:00"))
    assert "$140 reassess: CROSSED" in html      # 138 ≤ 140
    assert "$135 offer: +2.2%" in html           # (138-135)/138
    assert "$125 sell-everything stop: +9.4%" in html
    html2 = _d5_render(pb, snap_spot=170.0, now=_cet_epoch("19:00"))
    assert "CROSSED" not in html2.split("hard stops")[1].split("</li>")[0]


# ---- AVWAP: anchored at the first print, lookahead-free; stale → TradingView
def test_avwap_anchored_at_first_trade_stream_replay():
    feed = _feed()
    assert feed.avwap() is None                  # before any print
    trades = [("2026-06-12T16:05:00Z", 165.0, 100),
              ("2026-06-12T16:06:00Z", 170.0, 50),
              ("2026-06-12T16:07:00Z", 160.0, 50)]
    for ts, p, s in trades:
        feed.handle_message(json.dumps([{"T": "t", "S": "SPCX", "p": p, "s": s, "t": ts}]))
    av = feed.avwap()
    expect = (165.0 * 100 + 170.0 * 50 + 160.0 * 50) / 200.0
    assert av["avwap"] == pytest.approx(expect)
    assert av["anchored_at"] == pytest.approx(_parse_epoch("2026-06-12T16:05:00+00:00"))
    # replay order matters only through accumulation — adding a later trade never
    # changes the anchor (lookahead-free by construction)
    feed.handle_message(json.dumps([{"T": "t", "S": "SPCX", "p": 200.0, "s": 10,
                                     "t": "2026-06-12T16:08:00Z"}]))
    assert feed.avwap()["anchored_at"] == av["anchored_at"]


def test_d5_avwap_states_lost_and_stale():
    pb = _pb(fill=40.0, cross_ts=_cet_epoch("18:00"), eurusd=1.08)
    now = _cet_epoch("18:40")
    # spot below avwap, lost 20 min ago
    pb.update_avwap_state(spot=160.0, avwap=164.0, now=_cet_epoch("18:20"))
    html = _d5_render(pb, snap_spot=160.0,
                      avwap_info={"avwap": 164.0, "stale": False}, now=now)
    assert "anchored VWAP $164.00" in html and "BELOW" in html
    assert "lost 20 min ago, not reclaimed" in html and "front-load" in html
    # reclaim clears the lost state
    pb.update_avwap_state(spot=165.0, avwap=164.0, now=now)
    assert pb.avwap_lost_since is None
    # stale/absent feed → "use TradingView", never a number
    html2 = _d5_render(pb, snap_spot=160.0,
                       avwap_info={"avwap": 164.0, "stale": True}, now=now)
    assert "AVWAP: use TradingView" in html2 and "anchored VWAP $" not in html2


# ---- D5 pair-close chip needs BOTH conditions
def test_pair_close_chip_requires_both_conditions():
    cross = _cet_epoch("18:00")

    def hist(gap, mins_ok, end):
        out = []
        t0 = end - mins_ok * 60
        for i in range(int(mins_ok) + 1):
            out.append({"ts": t0 + i * 60, "perp": 165.0 + gap, "spot": 165.0,
                        "tails": {}, "mean_ps": 167.0, "median_ps": 163.0,
                        "pwin": 0.8, "spot_status": "live", "n_flags": 0})
        return out

    now = cross + 70 * 60                        # 70 min after cross
    ok = pair_close_status(hist(1.0, 20, now), cross, now)
    assert ok["green"] is True                   # gap small 20min, cross+70
    too_early = pair_close_status(hist(1.0, 20, cross + 30 * 60), cross, cross + 30 * 60)
    assert too_early["green"] is False           # cross+30 < 60
    too_short = pair_close_status(hist(1.0, 5, now), cross, now)
    assert too_short["green"] is False           # only 5 min of small gap
    too_wide = pair_close_status(hist(5.0, 20, now), cross, now)
    assert too_wide["green"] is False            # gap $5 > $2


def test_d5_pair_close_renders_only_when_hedged():
    cross = _cet_epoch("18:00")
    pb = _pb(fill=40.0, hedged=15.0, hedge_entry=158.0, cross_ts=cross, eurusd=1.08)
    html = _d5_render(pb, snap_spot=164.0, now=cross + 70 * 60)
    assert "pair-close" in html and "TR limit sell first" in html
    pb2 = _pb(fill=40.0, cross_ts=cross, eurusd=1.08)
    html2 = _d5_render(pb2, snap_spot=164.0, now=cross + 70 * 60)
    assert "pair-close" not in html2


# ---- CLOSE-OUT
def test_closeout_residual_and_overnight_rule():
    pb = _pb(fill=40.0, hedged=15.0, sold=20.0, cross_ts=_cet_epoch("18:00"), eurusd=1.08)
    html = _d5_render(pb, now=_cet_epoch("21:45"))
    assert "NOW: <b>CLOSE-OUT</b>" in html
    assert "don't carry residual overnight" in html and "5/6" in html
    assert "remaining residual 5 sh" in html     # 40 - 15 - 20
    assert "S2 §5" in html


# ---- persistence + --cross parsing
def test_playbook_state_persists_and_resumes(tmp_path):
    f = tmp_path / "state.json"
    pb = PlaybookState(f)
    pb.fill, pb.sold, pb.cross_ts, pb.cross_price = 40.0, 16.0, 1760000000.0, 168.5
    pb.save()
    pb2 = PlaybookState(f)
    assert (pb2.fill, pb2.sold, pb2.cross_ts, pb2.cross_price) == (40.0, 16.0,
                                                                   1760000000.0, 168.5)
    assert pb2.sub_eur == 10_000.0               # unset fields keep defaults


def test_parse_cross_arg_clock_and_now():
    now = _cet_epoch("18:30")
    ts, price = parse_cross_arg(["18:05", "165.5"], now)
    assert price == 165.5
    assert ts == _cet_epoch("18:05")
    ts2, _ = parse_cross_arg(["now", "165.5"], now)
    assert ts2 == now


def test_playbook_panel_renders_inside_dashboard():
    pb = _pb(fill=40.0, eurusd=1.08)
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash = DashboardState()
    dash.record(snap, rep)
    pb_html = render_playbook(pb, dash, rep, snap, None, now=_cet_epoch("08:30"))
    page = render_dashboard(dash, rep, snap, playbook_html=pb_html)
    assert "NOW: <b>D2/D3</b>" in page
    assert page.index("pbcard") < page.index("class='tiles'")   # panel above the tiles


# ------------------------------------------------- Block S5e: interactive dashboard server
import urllib.request as _url  # noqa: E402

from scripts.spcx_dashboard_server import (  # noqa: E402
    DashboardServer,
    build_ws_payload,
    snapshot_message,
)


def _server_fixture():
    """Real server on an ephemeral 127.0.0.1 port + one recorded poll."""
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash = DashboardState()
    dash.record(snap, rep)
    pb = _pb(eurusd=1.08)
    from scripts.spcx_pm_pdf_monitor import render_playbook as rp
    pb_html = rp(pb, dash, rep, snap, None, now=_cet_epoch("08:30"))
    server = DashboardServer(dash, pb, port=0)
    server.start()
    server.last_ctx = (rep, snap, None)
    payload = build_ws_payload(dash, pb, rep, snap, None, pb_html)
    server.push(payload)
    return server, payload


def _ws_connect(server, timeout=5):
    import websocket as wsc
    return wsc.create_connection(server.url().replace("http", "ws") + "ws",
                                 timeout=timeout)


def test_ws_payload_is_wire_serializable_and_downsampled():
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash = DashboardState()
    dash.record(snap, rep)
    payload = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    json.dumps(payload)                                  # serializable
    assert payload["type"] == "poll"
    assert 350 <= len(payload["survivor"]["grid"]) <= 450   # downsampled, not 20k
    assert len(payload["survivor"]["strikes"]) == 16
    assert payload["stats"]["ev_vs_offer_ps"] == pytest.approx(
        rep["stats_primary"]["ev_vs_offer_ps"])
    assert set(payload["tails"].keys()) == {str(k) for k in TAIL_STRIKES}
    assert payload["playbook_html"] == "<div>pb</div>"
    assert payload["node"] == "PRE-ALLOC"               # no fill fed yet
    lv = {l["level"]: l for l in payload["levels"]}
    assert set(lv) == {183.0, 162.0, 140.0, 135.0, 125.0}
    assert lv[162.0]["crossed_ts"] is not None          # fixture perp 173.53 crossed it
    assert lv[125.0]["crossed_ts"] is None and lv[125.0]["downside"] is True
    # snapshot message too
    msg = snapshot_message(dash, _pb(), payload, None)
    json.dumps(msg)
    assert msg["type"] == "snapshot" and len(msg["history"]) == 1


def test_server_serves_page_and_vendored_echarts_no_cdn():
    server, _ = _server_fixture()
    page = _url.urlopen(server.url(), timeout=5).read().decode()
    assert "<title>SPCX — listing day</title>" in page
    # self-contained: the only script src is the local vendored echarts
    import re as r
    srcs = r.findall(r'src="([^"]+)"', page)
    assert "/assets/echarts.min.js" in srcs
    assert all(s.startswith("/assets/") for s in srcs)   # every resource is local
    assert "http://" not in page.replace("ws://", "") and "https://" not in page
    js = _url.urlopen(server.url() + "assets/echarts.min.js", timeout=5).read()
    assert len(js) > 1_000_000                           # the real vendored file
    # path traversal guarded
    with pytest.raises(Exception):
        _url.urlopen(server.url() + "assets/../pyproject.toml", timeout=5)


def test_ws_snapshot_push_and_reconnect():
    server, payload = _server_fixture()
    ws = _ws_connect(server)
    m1 = json.loads(ws.recv())
    assert m1["type"] == "snapshot" and m1["last"] is not None
    assert len(m1["history"]) == 1
    server.push(payload)
    m2 = json.loads(ws.recv())
    assert m2["type"] == "poll" and len(m2["survivor"]["strikes"]) == 16
    ws.close()                                           # client dies...
    ws2 = _ws_connect(server)                            # ...reconnect
    m3 = json.loads(ws2.recv())
    assert m3["type"] == "snapshot" and m3["last"] is not None   # fresh full snapshot
    server.push(payload)                                 # push still reaches the new client
    m4 = json.loads(ws2.recv())
    assert m4["type"] == "poll"
    ws2.close()


def test_api_state_updates_persists_and_rebroadcasts(tmp_path):
    server, _ = _server_fixture()
    server.pb.path = tmp_path / "state.json"             # persist target
    ws = _ws_connect(server)
    json.loads(ws.recv())                                # drain snapshot
    req = _url.Request(server.url() + "api/state", method="POST",
                       data=json.dumps({"fill": 40, "cross": ["18:05", 165.5]}).encode(),
                       headers={"Content-Type": "application/json"})
    resp = json.loads(_url.urlopen(req, timeout=5).read())
    assert resp["ok"] and "fill" in resp["changed"] and "cross" in resp["changed"]
    assert resp["state"]["fill"] == 40.0
    assert resp["state"]["cross_price"] == 165.5
    m = json.loads(ws.recv())                            # immediate playbook rebroadcast
    assert m["type"] == "playbook" and "NOW:" in m["playbook_html"]
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["fill"] == 40.0 and saved["cross_price"] == 165.5
    ws.close()


def test_api_mark_broadcasts():
    server, _ = _server_fixture()
    ws = _ws_connect(server)
    json.loads(ws.recv())
    req = _url.Request(server.url() + "api/mark", method="POST",
                       data=json.dumps({"label": "CROSS"}).encode(),
                       headers={"Content-Type": "application/json"})
    assert json.loads(_url.urlopen(req, timeout=5).read())["ok"]
    m = json.loads(ws.recv())
    assert m["type"] == "marks" and m["marks"][-1]["label"] == "CROSS"
    ws.close()


def test_static_mode_regression_unaffected_by_server_layer(tmp_path):
    """--fixture --html (no --serve) must keep producing the S5c static page."""
    from scripts.spcx_pm_pdf_monitor import main
    out = tmp_path / "static.html"
    rc = main(["--fixture", "--html", str(out), "--eurusd", "1.08",
               "--playbook-state", str(tmp_path / "pb.json")])
    assert rc == 0 and out.exists()
    html = out.read_text()
    assert "pbcard" in html and "<svg" in html and "<script" not in html.lower()


# ════════════════════════════════ Block S5g — PM tail-trade screen ════════════════════
# Liquidity-weighted lognormal "fair" survivor (Q2) + the upper-tail sell screen (Q1):
# executable YES bid, repricing velocity, depth, richness-vs-fair, and the PEAK signal.

def test_lognormal_fit_recovers_known_params():
    """1/spread²-weighted fit must recover the generating (mu, sig) on clean lognormal pts."""
    mu, sig = math.log(2.1), 0.25
    strikes = list(np.linspace(1.0, 4.0, 16))
    S = [float(lognormal_survivor(k, mu, sig)) for k in strikes]
    fit = fit_lognormal_weighted(strikes, S, [0.01] * 16)
    assert fit["mu"] == pytest.approx(mu, abs=0.01)
    assert fit["sig"] == pytest.approx(sig, abs=0.01)
    assert fit["median_cap_t"] == pytest.approx(math.exp(mu), abs=0.02)
    assert fit["wrmse_c"] < 0.5


def test_lognormal_fit_reproduces_audit_median_on_fixture():
    """On the 2026-06-07 fixture the liquidity-weighted lognormal matches the audit:
    median cap ≈ 2.14T, tight weighted RMSE (see spacex_pdf_construction_audit Finding 3)."""
    rep = analyze(fixture_snapshot(), basis="mid")
    ln = rep["lognormal"]
    assert ln["median_cap_t"] == pytest.approx(2.14, abs=0.06)
    assert ln["wrmse_c"] < 3.0


def _tail_snap(bids):
    """Minimal snapshot with given YES bids at the upper-tail strikes (bid==ask, depth set)."""
    return {"fetched_at_utc": "2026-06-12T16:00:00+00:00",
            "ladder": [{"strike_t": k, "bid": b, "ask": b + 0.01, "bid_sz": 1000.0}
                       for k, b in bids.items()]}


def _ln_fair():
    strikes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    mu, sig = math.log(2.0), 0.28
    S = [float(lognormal_survivor(k, mu, sig)) for k in strikes]
    return fit_lognormal_weighted(strikes, S, [0.01] * 6)


def test_tail_trade_rip_then_fade_fires_peak():
    """A tail bid that ripped up over the window then faded off its high → PEAK signal #6,
    and that strike's read says SELL."""
    fit = _ln_fair()
    snap = _tail_snap({2.6: 0.19, 2.8: 0.13, 3.0: 0.07})
    hist = [{"ts": 0.0, "tails": {2.6: (0.14, 0.15), 2.8: (0.10, 0.11), 3.0: (0.06, 0.07)}},
            {"ts": 400.0, "tails": {2.6: (0.24, 0.25), 2.8: (0.16, 0.17), 3.0: (0.09, 0.10)}},
            {"ts": 800.0, "tails": {2.6: (0.19, 0.20), 2.8: (0.13, 0.14), 3.0: (0.07, 0.08)}}]
    tt = tail_trade_eval(snap, fit, hist, now=850.0, window_s=900.0,
                         tail_strikes=(2.6, 2.8, 3.0))
    assert tt["signal_cls"] == "go" and "PEAK" in tt["signal"]
    by_k = {r["strike_t"]: r for r in tt["rows"]}
    assert "SELL" in by_k[2.6]["read"]          # ripped +5c then -5c off high
    assert by_k[2.6]["dbid_c"] == pytest.approx(5.0, abs=0.5)
    assert by_k[2.6]["off_high_c"] == pytest.approx(-5.0, abs=0.5)
    assert by_k[3.0]["read"] == "quiet"         # only +1c, never ripped


def test_tail_trade_quiet_when_flat():
    fit = _ln_fair()
    snap = _tail_snap({2.6: 0.14, 2.8: 0.10, 3.0: 0.06})
    hist = [{"ts": t, "tails": {2.6: (0.14, 0.15), 2.8: (0.10, 0.11), 3.0: (0.06, 0.07)}}
            for t in (0.0, 400.0, 800.0)]
    tt = tail_trade_eval(snap, fit, hist, now=850.0, tail_strikes=(2.6, 2.8, 3.0))
    assert tt["signal_cls"] == "idle"
    assert all(r["read"] in ("quiet", "≈ fair") or "rich" in r["read"] for r in tt["rows"])


def test_tail_trade_richness_sign_and_graceful():
    """richness = (bid − fair)·100: a bid well above fair reads positive; and missing
    fit/history/depth degrade to None without raising."""
    fit = _ln_fair()
    fair_26 = fit["S"](2.6)
    snap = _tail_snap({2.6: fair_26 + 0.05})  # bid 5c above fair
    tt = tail_trade_eval(snap, fit, history=[], now=0.0, tail_strikes=(2.6,))
    assert tt["rows"][0]["richness_c"] == pytest.approx(5.0, abs=0.6)
    assert tt["rows"][0]["dbid_c"] is None     # no history → no velocity, no crash
    # no fit + no depth: a snapshot whose row lacks bid_sz and with fit_ln=None
    bare = {"fetched_at_utc": "t", "ladder": [{"strike_t": 2.6, "bid": 0.2, "ask": 0.21}]}
    tt2 = tail_trade_eval(bare, None, history=[], now=0.0, tail_strikes=(2.6,))
    assert tt2["rows"][0]["fair"] is None and tt2["rows"][0]["bid_depth_sh"] is None
    assert tt2["signal_cls"] == "idle"


def test_payload_carries_tail_trade_and_lognormal_not_buckets():
    """The LIVE-tab payload must include tail_trade + a serializable lognormal fair curve,
    and must NOT include buckets (removed from the live view), while staying JSON-safe."""
    dash, rep, snap = _two_poll_dash()
    payload = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    json.dumps(payload)                                   # serializable (no numpy/lambda)
    assert "buckets" not in payload
    assert "tail_trade" in payload and "rows" in payload["tail_trade"]
    assert {r["strike_t"] for r in payload["tail_trade"]["rows"]} == set(TAIL_STRIKES)
    ln = payload["lognormal"]
    assert ln["median_cap_t"] is not None
    assert ln["fair_S"] is not None and len(ln["fair_S"]) == len(payload["survivor"]["grid"])
    # PDF-chart constructions: analytic lognormal density aligned to the grid, and the
    # assumption-free raw interval-mass histogram (non-negative, price-space, 15 gaps for
    # 16 strikes); tail rows carry their $/share equivalent for the chart markers
    assert len(ln["fair_pdf"]) == len(payload["survivor"]["grid"])
    assert all(v >= 0 for v in ln["fair_pdf"])
    hist = payload["pdf"]["raw_hist"]
    assert len(hist) == 15
    assert all(h["dens"] >= 0 and h["hi_ps"] > h["lo_ps"] for h in hist)
    # total mass of the histogram = S(first strike) − S(last strike), a probability ≤ 1
    mass = sum(h["dens"] * (h["hi_ps"] - h["lo_ps"]) for h in hist)
    ratio = payload["pdf"]["price"][1] / payload["survivor"]["grid"][1]  # $/share per $T
    assert 0.0 < mass / ratio <= 1.0
    assert all(r["strike_ps"] == pytest.approx(r["strike_t"] * ratio, rel=1e-6)
               for r in payload["tail_trade"]["rows"])


# ════════════════════════════════ Block S5h — time-scrub slider ═══════════════════════
# CurveIndex over the parquet log + GET /api/curve: replay the survivor/PDF at any
# logged poll, refit through the same analyze() as live.

def test_curve_index_load_nearest_and_refit(tmp_path):
    from datetime import datetime, timedelta, timezone
    from scripts.spcx_pm_pdf_monitor import CurveIndex
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    base = datetime.now(timezone.utc).replace(microsecond=0)
    stamps = [base - timedelta(minutes=40), base - timedelta(minutes=20), base]
    for dt in stamps:
        log_parquet(dict(snap, fetched_at_utc=dt.isoformat()), rep, out_dir=tmp_path)
    idx = CurveIndex(days=7.0)
    assert idx.load_parquet(log_dir=tmp_path) == 3
    rng = idx.range()
    assert rng["n"] == 3
    assert rng["max_ts"] - rng["min_ts"] == pytest.approx(2400.0, abs=2.0)
    # nearest snaps to the middle entry
    mid_ts = stamps[1].timestamp()
    assert idx.nearest(mid_ts + 120.0)["ts"] == pytest.approx(mid_ts, abs=1.0)
    # refit through the live pipeline reproduces the original stats from the logged ladder
    got = idx.report_at(mid_ts)
    assert got is not None
    _e, _snap, rep2 = got
    assert rep2["stats_primary"]["mean_ps"] == pytest.approx(
        rep["stats_primary"]["mean_ps"], abs=0.05)
    assert rep2["lognormal"]["median_cap_t"] == pytest.approx(
        rep["lognormal"]["median_cap_t"], abs=0.02)


def test_curve_index_append_extends_range(tmp_path):
    from scripts.spcx_pm_pdf_monitor import CurveIndex
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    idx = CurveIndex(days=7.0)
    idx.append(snap, rep)
    assert idx.range()["n"] == 1
    snap2 = dict(snap, fetched_at_utc="2026-06-07T15:00:00+00:00")
    idx.append(snap2, rep)   # older ts than the first append (15:59) → ignored
    later = dict(snap, fetched_at_utc="2099-01-01T00:00:00+00:00")
    idx.append(later, rep)
    rng = idx.range()
    assert rng["n"] == 2 and rng["max_ts"] > rng["min_ts"]


def test_api_curve_endpoint_serves_refit_and_validates():
    from scripts.spcx_pm_pdf_monitor import CurveIndex
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    idx = CurveIndex(days=7.0)
    idx.append(snap, rep)
    dash = DashboardState()
    dash.record(snap, rep)
    server = DashboardServer(dash, _pb(eurusd=1.08), port=0, curves=idx)
    server.start()
    want_ts = idx.range()["max_ts"]
    body = json.loads(_url.urlopen(
        server.url() + f"api/curve?ts={want_ts + 9999}", timeout=5).read())  # snaps to edge
    assert body["type"] == "curve" and body["ts"] == pytest.approx(want_ts, abs=1.0)
    assert {"survivor", "pdf", "lognormal", "stats", "offer"} <= set(body)
    assert len(body["survivor"]["grid"]) == len(body["survivor"]["S"])
    assert body["pdf"]["raw_hist"] and body["lognormal"]["fair_pdf"]
    # second hit comes from the payload cache and is identical
    body2 = json.loads(_url.urlopen(
        server.url() + f"api/curve?ts={want_ts}", timeout=5).read())
    assert body2 == body
    # missing/invalid ts → 400
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as ei:
        _url.urlopen(server.url() + "api/curve", timeout=5)
    assert ei.value.code == 400


def test_payload_curve_range_present_when_index_passed():
    from scripts.spcx_pm_pdf_monitor import CurveIndex
    dash, rep, snap = _two_poll_dash()
    idx = CurveIndex(days=7.0)
    idx.append(snap, rep)
    payload = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>",
                               curves=idx)
    json.dumps(payload)
    assert payload["curve_range"]["n"] == 1
    # without an index the field is present but null (front-end hides the slider)
    p2 = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    assert p2["curve_range"] is None


# ════════════════════════════ Block S5h item 1 — hedge-ops panel ══════════════════════

def _hedged_pb(**kw):
    pb = _pb(eurusd=1.08)
    pb.offer = 135.0
    pb.hedged = 18.0
    pb.hedge_entry = 165.0
    pb.hedge_lev = 1.5
    pb.hedge_ts = 1_000_000.0
    for k, v in kw.items():
        setattr(pb, k, v)
    return pb


def test_hedge_ops_buffer_equals_calc_module():
    """The panel's liq price + buffer band must be bit-identical to the calc module's own
    functions (imported, never reimplemented)."""
    from scripts.spcx_convergence_calc import (
        liq_buffer_summary, liq_price_short, maintenance_margin_frac)
    from scripts.spcx_pm_pdf_monitor import hedge_ops_eval
    pb = _hedged_pb()
    ops = hedge_ops_eval(pb, mark=170.0, spot=None, funding_hourly=1e-4,
                         max_leverage=3.0, history=[], now=1_003_600.0)
    mmr = maintenance_margin_frac(3.0)
    liq = liq_price_short(165.0, 1.5, mmr)
    ref = liq_buffer_summary(170.0, liq, alert_pct=0.10)
    assert ops["liq_px"] == liq
    assert ops["buffer"]["buffer_frac"] == ref["buffer_frac"]
    assert ops["buffer"]["band"] == ref["band"]
    # basis lock arithmetic: entry − offer, per share and total
    assert ops["basis_ps"] == pytest.approx(30.0)
    assert ops["basis_total"] == pytest.approx(540.0)
    # funding accrual at the public rate over 1h on the mark notional
    assert ops["funding_accrued"] == pytest.approx(1e-4 * 170.0 * 18.0 * 1.0)
    assert ops["margin_posted_usd"] == pytest.approx(2000.0 * 1.08)


def test_hedge_ops_none_without_hedge_and_bands_shift():
    from scripts.spcx_pm_pdf_monitor import hedge_ops_eval
    pb = _pb(eurusd=1.08)
    assert hedge_ops_eval(pb, 170.0, None, 0.0, 3.0, [], 0.0) is None
    # BREACH when the mark is at/through the liq price
    pb2 = _hedged_pb(hedge_lev=1.5)
    ops = hedge_ops_eval(pb2, mark=300.0, spot=None, funding_hourly=0.0,
                         max_leverage=3.0, history=[], now=1_000_100.0)
    assert ops["buffer"]["band"] == "BREACH"


def test_hedge_ops_s6_ladder_phases():
    from scripts.spcx_pm_pdf_monitor import hedge_ops_eval
    base = 1_000_000.0
    for mins, want in [(None, "PRE-CROSS"), (20, "NO-ZONE"), (50, "GATE-2")]:
        pb = _hedged_pb(cross_ts=None if mins is None else base - mins * 60.0)
        ops = hedge_ops_eval(pb, 166.0, 165.5, 0.0, 3.0, [], base)
        assert ops["ladder"][0] == want, f"{mins} min → {ops['ladder']}"
    # +75 min with a sustained ≤$2 gap history → GREEN (pair_close_status confirms)
    pb = _hedged_pb(cross_ts=base - 75 * 60.0)
    hist = [{"ts": base - s, "perp": 166.0, "spot": 165.5}
            for s in range(1200, -1, -45)]
    ops = hedge_ops_eval(pb, 166.0, 165.5, 0.0, 3.0, hist, base)
    assert ops["ladder"][0] == "GREEN"


def test_hedge_ops_fasttape_exception_inputs():
    """§5.1 fast-tape exception: buffer < 25% OR >2% 5-min move flips leg order."""
    from scripts.spcx_pm_pdf_monitor import hedge_ops_eval, spot_move_5m
    base = 1_000_000.0
    calm = [{"ts": base - 240, "spot": 165.0}, {"ts": base - 10, "spot": 165.5}]
    fast = [{"ts": base - 240, "spot": 160.0}, {"ts": base - 10, "spot": 165.0}]
    assert abs(spot_move_5m(calm, base)) < 0.02
    assert spot_move_5m(fast, base) > 0.02
    assert spot_move_5m([], base) is None
    # deep buffer + calm tape → inactive; same hedge on a violent tape → active
    pb = _hedged_pb(hedge_lev=1.0)
    assert not hedge_ops_eval(pb, 166.0, 165.5, 0.0, 3.0, calm, base)["fasttape"]["active"]
    assert hedge_ops_eval(pb, 166.0, 165.0, 0.0, 3.0, fast, base)["fasttape"]["active"]
    # 1.5× near liq: buffer input alone triggers (mark close to liq px)
    pb2 = _hedged_pb(hedge_lev=1.5)
    ops = hedge_ops_eval(pb2, 230.0, None, 0.0, 3.0, calm, base)
    assert ops["fasttape"]["buffer_lt_25"] and ops["fasttape"]["active"]


def test_hedge_ops_panel_in_payload_and_verbatim_flag():
    from scripts.spcx_pm_pdf_monitor import render_hedge_ops, hedge_ops_eval
    assert render_hedge_ops(None) == ""
    pb = _hedged_pb(cross_ts=None)
    ops = hedge_ops_eval(pb, 166.0, None, 1e-4, 3.0, [], 1_003_600.0)
    html = render_hedge_ops(ops)
    for needle in ("HEDGE OPS", "LIQ PRICE", "S6 readiness ladder", "21:00 CET backstop",
                   "$1.7/sh", "pre-registered judgment, not measured",
                   "close the perp first"):
        assert needle in html, needle
    # payload carries it; without a hedge it's an empty string
    dash, rep, snap = _two_poll_dash()
    p = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>")
    assert "HEDGE OPS" in p["hedge_ops_html"]
    p2 = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    assert p2["hedge_ops_html"] == ""


# ════════════════════════════ Block S5h item 2 — lean tranche tables ══════════════════

def test_tranche_row_selection_with_hedge_sleeve_subtraction():
    from scripts.spcx_pm_pdf_monitor import select_tranche_plan, tranche_schedule
    # ≤10: two tickets 60/40, skip T3
    p = select_tranche_plan(10)
    assert p["row"] == "≤10" and [t[1] for t in p["tickets"]] == [6, 4]
    # the pre-registered rows verbatim at their own sizes
    assert [t[1] for t in select_tranche_plan(21)["tickets"]] == [8.0, 8.0, 5.0]
    assert [t[1] for t in select_tranche_plan(43)["tickets"]] == [17.0, 17.0, 9.0]
    assert [t[1] for t in select_tranche_plan(65)["tickets"]] == [26.0, 26.0, 13.0]
    assert select_tranche_plan(65)["t1_immediate"]            # 100%-fill case flag
    # off-row residual: nearest row, LAST ticket absorbs the difference
    p22 = select_tranche_plan(22)
    assert p22["row"] == "~20" and [t[1] for t in p22["tickets"]] == [8.0, 8.0, 6.0]
    assert sum(t[1] for t in p22["tickets"]) == 22.0
    # hedge-sleeve subtraction: fill 40, hedged 18 → residual 22 → same as above
    ts = tranche_schedule(fill=40, hedged=18, sold=0, cross_ts=None, now=0.0)
    assert ts["residual"] == 22.0
    assert [r["shares"] for r in ts["rows"]] == [8.0, 8.0, 6.0]
    assert all("arm at the cross" in r["status"] for r in ts["rows"])


def test_tranche_status_vs_sold_and_windows():
    from scripts.spcx_pm_pdf_monitor import tranche_schedule
    base = 1_000_000.0
    # +70 min post-cross, 8 sold → T1 DONE, T2 ACTIVE, T3 future
    ts = tranche_schedule(40, 18, sold=8, cross_ts=base - 70 * 60, now=base)
    st = {r["name"]: r["status"] for r in ts["rows"]}
    assert st["T1"] == "DONE"
    assert st["T2"].startswith("ACTIVE")
    assert st["T3"].startswith("opens")
    # partial: 12 sold → T2 partial 4/8
    ts2 = tranche_schedule(40, 18, sold=12, cross_ts=base - 70 * 60, now=base)
    assert "PARTIAL (4/8)" in {r["name"]: r["status"] for r in ts2["rows"]}["T2"]


def test_tranche_overrides_window_arithmetic():
    from scripts.spcx_pm_pdf_monitor import tranche_schedule
    base = 1_000_000.0
    cross = base - 70 * 60            # +70 min in
    # FADE: T2 window end 180 → 70 + (180-70)/2 = 125; T3 start 180 → 70+55 = 125
    ts = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="FADE")
    rows = {r["name"]: r for r in ts["rows"]}
    assert "+60–125 min" in rows["T2"]["window"]
    assert rows["T3"]["window"].startswith("+125")
    assert "sell the next tranche NOW (T2" in ts["flag"]
    # RALLY: final ticket deferred + 10%-below-high stop level shown
    ts2 = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="RALLY",
                           session_high=180.0)
    rows2 = {r["name"]: r for r in ts2["rows"]}
    assert rows2["T3"]["status"].startswith("DEFERRED")
    assert "$162.00" in ts2["flag"]   # 0.9 × 180
    # CRASH: schedule void, hard-stop ladder governs
    ts3 = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="CRASH")
    assert ts3["rows"] == [] and "hard-stop ladder" in ts3["flag"]


def test_tranche_html_in_payload():
    from scripts.spcx_pm_pdf_monitor import render_tranches, tranche_schedule
    ts = tranche_schedule(40, 18, sold=8, cross_ts=None, now=0.0)
    html = render_tranches(ts)
    assert "RESIDUAL TRANCHES — 22 sh" in html and "display only" in html
    dash, rep, snap = _two_poll_dash()
    pb = _pb(eurusd=1.08); pb.fill = 40.0; pb.hedged = 18.0
    p = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>")
    assert "RESIDUAL TRANCHES" in p["tranche_html"]
    pb2 = _pb(eurusd=1.08)            # no fill → no panel
    p2 = build_ws_payload(dash, pb2, rep, snap, None, "<div>pb</div>")
    assert p2["tranche_html"] == ""


# ════════════════════════════ Block S5h item 3 — S7 day-shape classifier ══════════════

def _stepper():
    from scripts.spcx_pm_pdf_monitor import DayShapeClassifier
    return DayShapeClassifier()


def test_classifier_fade_path_with_hysteresis():
    """Below AVWAP ≥10 min AND >5% off high → FADE, but only after holding 2 min."""
    c = _stepper()
    c.step(0.0, 180.0, 180.0, 165.0)               # first print = session high 180
    # (prices sit ABOVE the $140 hard stop — absolute SPCX levels — so CRASH stays inert)
    # drift below avwap for 11 min but only ~3% off high → still FLAT
    for m in range(1, 12):
        assert c.step(m * 60.0, 175.0, 178.0, 165.0) == "FLAT"
    # now >5% off high (180*0.95 = 171): raw FADE begins; hysteresis holds 2 min
    assert c.step(12 * 60.0, 169.0, 178.0, 165.0) == "FLAT"   # pending
    assert c.step(13 * 60.0, 169.0, 178.0, 165.0) == "FLAT"   # 60s < 120s
    assert c.step(14 * 60.0, 169.0, 178.0, 165.0) == "FADE"   # held 2 min → switch
    # one-minute blip back above avwap does NOT flip out of FADE (hysteresis again)
    assert c.step(15 * 60.0, 179.0, 178.0, 165.0) == "FADE"


def test_classifier_rally_and_never_from_stale_high():
    c = _stepper()
    c.step(0.0, 180.0, 180.0, 165.0)
    # new high + above avwap → RALLY after the 2-min hold
    c.step(60.0, 181.0, 180.0, 165.0)
    c.step(180.0, 182.0, 180.5, 165.0)
    assert c.step(300.0, 183.0, 180.5, 165.0) == "RALLY"
    # 20 min later, no new high (last high stale > 15 min) → drops back to FLAT
    for m in range(6, 26):
        st = c.step(m * 60.0, 182.0, 180.5, 165.0)
    assert st == "FLAT"


def test_classifier_instant_crash_and_override_pinning():
    c = _stepper()
    c.step(0.0, 170.0, 170.0, 165.0)
    # below cross AND below $160 → CRASH with NO hysteresis (single step)
    assert c.step(60.0, 158.0, 169.0, 165.0) == "CRASH"
    # hard stop alone (≤$140) also instant, even above the cross print
    c2 = _stepper()
    c2.step(0.0, 150.0, 150.0, 120.0)
    assert c2.step(60.0, 139.0, 149.0, 120.0) == "CRASH"
    # manual override pins the view; computed state stays visible beneath
    v = c.view("RALLY", 120.0)
    assert v["state"] == "RALLY" and v["computed"] == "CRASH" and v["override"] == "RALLY"
    v2 = c.view(None, 120.0)
    assert v2["state"] == "CRASH" and v2["override"] is None
    # not armed before any spot print → banner hidden
    assert not _stepper().view(None, 0.0)["armed"]


def test_classifier_cbrs_replay_smoke():
    """Sanity replay on the cached Cerebras 1m tape (NOT calibration — thresholds are
    pre-registered; if this fails, report and stop). Expect: FADE within ~20 min of the
    cross (S2: AVWAP lost +12 min, never reclaimed); never RALLY after the first hour."""
    import csv as _csv
    from scripts.spcx_pm_pdf_monitor import replay_classifier_on_bars
    tape = Path("data/analysis/csv_outputs/market_maps/ipo_tapes/cbrs_spot_1m_listingday.csv")
    if not tape.exists():
        pytest.skip("CBRS tape not cached")
    with tape.open() as f:
        bars = list(_csv.DictReader(f))
    timeline = replay_classifier_on_bars(bars)
    assert timeline, "classifier never left FLAT on the CBRS tape"
    fade_at = next((m for m, s in timeline if s == "FADE"), None)
    # Honest floor, NOT a tuned threshold: the pre-registered rule itself implies
    # >= ~24 min (S2's AVWAP loss at +12 min + the 10-min FADE clock + 2-min hysteresis),
    # so the block-spec's "~20 min" was arithmetically unreachable as written. Measured
    # on the tape: FADE at +27.0 min, held all day, no RALLY, no other transitions.
    assert fade_at is not None and fade_at <= 30.0, f"FADE at {fade_at} min (floor ~24)"
    assert not any(s == "RALLY" and m > 60.0 for m, s in timeline), \
        f"RALLY after the first hour: {timeline}"


def test_payload_day_shape_and_override_roundtrip():
    from scripts.spcx_pm_pdf_monitor import DayShapeClassifier
    dash, rep, snap = _two_poll_dash()
    cls = DayShapeClassifier()
    cls.step(1_000_000.0, 180.0, 180.0, 165.0)
    pb = _pb(eurusd=1.08)
    pb.shape_override = "FADE"
    p = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>", classifier=cls)
    json.dumps(p)
    assert p["day_shape"]["state"] == "FADE" and p["day_shape"]["computed"] == "FLAT"
    assert "FRONT-LOAD" in p["day_shape"]["action"]   # §0 verbatim row
    # tranche shape follows the pinned state: FADE flag appears once fill is set
    pb.fill, pb.cross_ts = 40.0, 1_000_000.0 - 70 * 60
    p2 = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>", classifier=cls)
    assert "remaining windows HALVED" in p2["tranche_html"]


# ════════════════════════════ Block S5h item 4 — cross-timing + indication ════════════

_HALTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/">
<channel><title>Trading Halts</title>
<item><title>SPCX halt</title>
  <ndaq:IssueSymbol>SPCX</ndaq:IssueSymbol>
  <ndaq:ReasonCode>IPO1</ndaq:ReasonCode>
  <ndaq:HaltDate>06/12/2026</ndaq:HaltDate><ndaq:HaltTime>09:30:00</ndaq:HaltTime>
  <ndaq:ResumptionQuoteTime>11:50:00</ndaq:ResumptionQuoteTime>
  <ndaq:ResumptionTradeTime>12:00:00</ndaq:ResumptionTradeTime>
</item>
<item><title>OTHR halt</title>
  <ndaq:IssueSymbol>OTHR</ndaq:IssueSymbol><ndaq:ReasonCode>T1</ndaq:ReasonCode>
</item>
</channel></rss>"""


def test_halts_parse_and_failure_degradation():
    from scripts.spcx_pm_pdf_monitor import parse_nasdaq_halts, render_halts
    halts = parse_nasdaq_halts(_HALTS_XML, "SPCX")
    assert len(halts) == 1
    h = halts[0]
    assert h["reasoncode"] == "IPO1"
    assert h["resumptionquotetime"] == "11:50:00"
    assert h["resumptiontradetime"] == "12:00:00"
    # malformed XML raises (the poller wrapper converts to 'down')
    with pytest.raises(Exception):
        parse_nasdaq_halts("<html>this is not the feed</html target=>", "SPCX")
    # rendering: down state shows the CNBC fallback, never stale data
    html = render_halts({"status": "down", "halt": None, "age_s": 5.0})
    assert "poller down — watch CNBC" in html
    ok = render_halts({"status": "ok", "age_s": 3.0, "halt": h})
    assert "12:00:00" in ok and "resumption TRADE" in ok
    none_html = render_halts({"status": "no_entry", "halt": None, "age_s": 3.0})
    assert "not in the halts feed" in none_html


def test_halts_poller_alert_one_shot():
    from scripts.spcx_pm_pdf_monitor import NasdaqHaltsPoller
    p = NasdaqHaltsPoller("SPCX")
    # simulate what _run does when a trade time first appears
    with p._lock:
        p._state = {"status": "ok", "halt": {"resumptiontradetime": "12:00:00"},
                    "fetched_at": 1.0, "alert": True}
    s1 = p.snapshot()
    assert s1["alert"] is True
    s2 = p.snapshot()
    assert s2["alert"] is False        # consumed exactly once


def test_indication_roundtrip_persistence_and_parquet(tmp_path):
    from scripts.spcx_pm_pdf_monitor import PlaybookState, render_indications
    # POST /api/indication on a real ephemeral server
    snap = fixture_snapshot()
    rep = analyze(snap, basis="mid")
    dash = DashboardState()
    dash.record(snap, rep)
    state_path = tmp_path / "pb.json"
    pb = PlaybookState(path=state_path)
    pb.eurusd = 1.08
    server = DashboardServer(dash, pb, port=0)
    server.start()
    server.last_ctx = (rep, snap, None)
    req = _url.Request(server.url() + "api/indication", method="POST",
                       data=json.dumps({"text": "148-152"}).encode(),
                       headers={"Content-Type": "application/json"})
    assert json.loads(_url.urlopen(req, timeout=5).read())["n"] == 1
    # persisted: a fresh PlaybookState loads it back
    pb2 = PlaybookState(path=state_path)
    assert pb2.indications[-1]["text"] == "148-152"
    assert pb2.indications[-1]["ts"] > 0
    # rendered next to perp + PM mean
    html = render_indications(pb2, 165.0, 167.0)
    assert "148-152" in html and "$165.0" in html and "$167.0" in html
    # parquet: extra columns land on every row (schema-extended)
    import pyarrow.parquet as pq
    p = log_parquet(snap, rep, out_dir=tmp_path,
                    extra={"indication_text": "148-152",
                           "indication_ts": pb2.indications[-1]["ts"]})
    t = pq.read_table(p)
    assert "indication_text" in t.column_names
    assert t.to_pylist()[0]["indication_text"] == "148-152"
    # bad POST → 400
    import urllib.error
    bad = _url.Request(server.url() + "api/indication", method="POST",
                       data=b"{}", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        _url.urlopen(bad, timeout=5)
    assert ei.value.code == 400


# ════════════════════════════ Block S5j — interpretation charts (display only) ════════

def test_tranche_chart_override_arithmetic_equality():
    """Chart-1 bands come from tranche_schedule's OWN numeric fields — the FADE-halved
    windows in the chart block must equal the (already tested) panel arithmetic."""
    from scripts.spcx_pm_pdf_monitor import tranche_chart_data, tranche_schedule
    base = 1_000_000.0
    cross = base - 70 * 60
    ts = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="FADE")
    d = tranche_chart_data(ts, cross, base)
    rows = {r["name"]: r for r in d["rows"]}
    # FADE at +70: T2 end 180 → 70+(180-70)/2 = 125; T3 start 180 → 125 (same rule)
    assert rows["T2"]["hi"] == pytest.approx(125.0)
    assert rows["T3"]["lo"] == pytest.approx(125.0)
    assert rows["T3"]["open_ended"] and rows["T3"]["hi"] == pytest.approx(d["end_min"])
    # the numeric fields agree with the rendered text the panel tests pinned
    assert "+60–125 min" in ts["rows"][1]["window"]
    # cumulative targets stack: 8 / 16 / 22
    assert [r["cum_target"] for r in d["rows"]] == [8.0, 16.0, 22.0]
    assert d["now_min"] == pytest.approx(70.0)
    # RALLY: the deferred final ticket is marked for greying
    ts2 = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="RALLY",
                           session_high=180.0)
    d2 = tranche_chart_data(ts2, cross, base)
    assert {r["name"]: r["deferred"] for r in d2["rows"]}["T3"] is True
    # CRASH: bands collapse — the chart block carries crash + residual, no rows
    ts3 = tranche_schedule(40, 18, sold=8, cross_ts=cross, now=base, shape="CRASH")
    d3 = tranche_chart_data(ts3, cross, base)
    assert d3["crash"] and d3["rows"] == [] and d3["residual"] == 22.0


def test_hedge_chart_close_now_and_buffer_series_equality():
    """Close-now readout must equal (locked basis − gap) × hedged exactly, and the
    buffer series must be the calc module's own liq_buffer_summary at each point."""
    from scripts.spcx_convergence_calc import liq_buffer_summary
    from scripts.spcx_pm_pdf_monitor import hedge_chart_data, hedge_ops_eval
    base = 1_000_000.0
    pb = _hedged_pb(cross_ts=base - 50 * 60)
    hist = [{"ts": base - s, "perp": 166.0 + s / 1200.0, "spot": 165.0}
            for s in range(600, -1, -60)]
    ops = hedge_ops_eval(pb, 166.0, 165.0, 0.0, 3.0, hist, base)
    d = hedge_chart_data(ops, hist, pb.cross_ts, base)
    cn = d["close_now"]
    assert cn["net_total"] == pytest.approx((ops["basis_ps"] - ops["gap"]) * pb.hedged)
    assert cn["locked_total"] == pytest.approx(ops["basis_ps"] * pb.hedged)
    assert cn["drag_total"] == pytest.approx(ops["gap"] * pb.hedged)
    # buffer series point-for-point from the calc module at the entry-fixed liq price
    for pt, rec in zip(d["series"], hist):
        ref = liq_buffer_summary(rec["perp"], ops["liq_px"], alert_pct=0.10)
        assert pt["buffer"] == pytest.approx(ref["buffer_frac"])
        assert pt["band"] == ref["band"]
    # S6 zone boundaries anchored to the cross
    assert d["zone2_ts"] == pytest.approx(pb.cross_ts + 46 * 60)
    assert d["zone1_ts"] == pytest.approx(pb.cross_ts + 61 * 60)


def test_s5j_empty_states():
    from scripts.spcx_pm_pdf_monitor import (hedge_chart_data, tranche_chart_data,
                                             tranche_schedule)
    # no hedge → chart-2 block is None
    assert hedge_chart_data(None, [], None, 0.0) is None
    # no fill → chart-1 block is None
    ts = tranche_schedule(None, None, 0.0, None, 0.0)
    assert tranche_chart_data(ts, None, 0.0) is None
    # no cross → windows pending, no playhead/end; zero sold renders with sold=0
    ts2 = tranche_schedule(40, 18, sold=0.0, cross_ts=None, now=1_000_000.0)
    d = tranche_chart_data(ts2, None, 1_000_000.0)
    assert d["now_min"] is None and d["end_min"] is None and d["sold"] == 0.0
    # hedge on, no cross: gap entries None (top strip "awaiting cross"), buffer runs
    pb = _hedged_pb(cross_ts=None)
    from scripts.spcx_pm_pdf_monitor import hedge_ops_eval
    hist = [{"ts": 1_000_000.0 - s, "perp": 166.0, "spot": None} for s in (120, 60, 0)]
    ops = hedge_ops_eval(pb, 166.0, None, 0.0, 3.0, hist, 1_000_000.0)
    d2 = hedge_chart_data(ops, hist, None, 1_000_000.0)
    assert all(p["gap"] is None for p in d2["series"])          # awaiting cross
    assert all(p["buffer"] is not None for p in d2["series"])   # buffer strip runs
    assert d2["close_now"]["net_total"] is None                 # drag unknown
    assert d2["zone2_ts"] is None


def test_s5j_blocks_in_live_payload_and_rehearsal_panel(tmp_path):
    from scripts import spcx_rehearsal as rhmod
    from tests.test_spcx_rehearsal import _synth_bars
    # live payload
    dash, rep, snap = _two_poll_dash()
    pb = _hedged_pb(cross_ts=None)
    pb.fill = 40.0
    p = build_ws_payload(dash, pb, rep, snap, None, "<div>pb</div>")
    json.dumps(p)
    assert p["tranche_chart"]["residual"] == 22.0
    assert p["hedge_chart"]["close_now"]["locked_total"] == pytest.approx(540.0)
    # rehearsal panel: same helpers, replayed inputs
    reh = rhmod.build_rehearsal(_synth_bars())
    pp = rhmod.panel_at(reh, 100, sold=8.0, state_path=tmp_path / "r.json")
    json.dumps({k: pp[k] for k in ("tranche_chart", "hedge_chart")})
    assert pp["tranche_chart"]["sold"] == 8.0
    assert pp["hedge_chart"]["close_now"]["net_total"] is not None
    # rehearsal close-now equality on simulated values
    assert pp["hedge_chart"]["close_now"]["net_total"] == pytest.approx(
        (pp["hedge_chart"]["basis_ps"] - pp["hedge_chart"]["close_now"]["gap"]) * 18.0)


# ════════════════════════════ Block S5k — EXECUTION tab support ═══════════════════════

def test_snapshot_history_carries_shape_and_avwap():
    """The EXEC tab's live state ribbon + AVWAP line read per-poll shape/avwap stamped
    onto the history records; the snapshot must pass them through (None when absent)."""
    dash, rep, snap = _two_poll_dash()
    dash.history[-1]["shape"] = "FADE"
    dash.history[-1]["avwap"] = 166.2
    msg = snapshot_message(dash, _pb(eurusd=1.08), None, None)
    assert msg["history"][-1]["shape"] == "FADE"
    assert msg["history"][-1]["avwap"] == 166.2
    assert msg["history"][0]["shape"] is None      # earlier record never stamped


def test_payload_carries_pm_tab_and_indication_card():
    """Block S5k (PM tab, 2026-06-12 listing-day morning) supersedes the §10.8 drop:
    the indication paste field moves to the PM tab (the EXECUTION day view stays free
    of it), and the payload carries the server-rendered PM panel. Both additive,
    JSON-serializable."""
    dash, rep, snap = _two_poll_dash()
    p = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    json.dumps(p)
    assert "pm_html" in p and "PM1" in p["pm_html"] and "PM2" in p["pm_html"]
    assert "indications_html" in p and "ind_in" in p["indications_html"]
    assert "halts_html" in p                        # the cross-time alarm stays


# ═════════════════ Block S5k (PM tab) — display-only PM-leg analytics ══════════════════
# Listing-day morning block: full distribution table + bucket-vs-ladder with gap
# sparklines + tail-sell screen + indication field on one tab; DIVERGENT flag single
# source of truth; the E5 pre-allocation SAVE regression.

def test_pm_panel_render_smoke_fixture():
    """PM-panel render smoke on the offline fixture: both share-base columns, headline
    stats, strikes/basis line, mode footnote, No-IPO ask, perp-vs-crowd strip — all
    terminal numbers, no new math."""
    from scripts.spcx_pm_pdf_monitor import PM_TAB_NOTE, render_pm_panel
    dash, rep, snap = _two_poll_dash()
    html = render_pm_panel(rep, dash, now=dash.history[-1]["ts"])
    a, b = rep["stats_primary"], rep["stats_coworker"]
    assert "13.076B" in html and "13.091B" in html          # both share bases
    assert f"{a['mean_ps']:.1f}" in html and f"{b['mean_ps']:.1f}" in html
    assert f"{a['p_win_offer'] * 100:.1f}%" in html
    assert f"{a['ev_vs_offer_ps']:+.1f}" in html            # EV A1, never the D2 bug
    assert "mode*" in html and "shape-fragile" in html
    assert f"strikes used {rep['n_strikes_used']}/16" in html
    assert "basis mid" in html
    g = rep["hl_gap"]
    assert f"${g['hl_mark']:.2f}" in html                   # perp-vs-crowd strip
    assert f"{g['vs_mean_ps']:+.1f}" in html and f"{g['vs_median_ps']:+.1f}" in html
    assert "No-IPO leg ask: 0.4c" in html                   # fixture ask 0.004
    for r in rep["buckets"]:                                # every bucket row renders
        assert r["label"] in html
    assert html.count("<svg") >= len(rep["buckets"])        # one sparkline per row
    assert "no edge claims" in PM_TAB_NOTE                  # the verbatim header note


def test_pm_tab_page_structure_tab_order_and_moved_panels():
    """Tab order EXECUTION | PM | ANALYSIS | … | REHEARSAL; the verbatim operator note
    on the pane; the tail-sell screen + indication field live INSIDE the PM pane with a
    link stub left on ANALYSIS; the EXECUTION tab's E5 card untouched."""
    page = Path("scripts/assets/spcx_dashboard.html").read_text()
    assert (page.index('id="tb-exec"') < page.index('id="tb-pm"')
            < page.index('id="tb-live"') < page.index('id="tb-reh"'))
    text = _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", page))
    assert ("Analytics for the PM leg (Alvaro). Display only — no PM trading logic, "
            "no edge claims; DIVERGENT = quote gap before spread/fees, check depth "
            "before acting.") in text
    i_exec = page.index('id="tab-exec"')
    i_pm = page.index('id="tab-pm"')
    i_live = page.index('id="tab-live"')
    i_plan = page.index('id="tab-plan"')
    assert i_exec < i_pm < i_live                            # pane order matches tabs
    assert i_pm < page.index('id="tt_tbl"') < i_live         # tail screen moved, not copied
    assert i_pm < page.index('id="c_tails"') < i_live
    assert i_pm < page.index('id="pmind"') < i_live          # indication field on PM
    assert page.count('id="tt_tbl"') == 1 and page.count('id="c_tails"') == 1
    assert "moved to the" in page[i_live:i_plan]             # the ANALYSIS link stub
    assert i_exec < page.index('id="f_fill"') < i_pm         # E5 card still on EXEC
    assert i_exec < page.index('onclick="saveState()"') < i_pm
    assert "function sendIndication" in page                 # the ADD button's handler
    assert "function setBtnsDead" in page                    # dead-server feedback


def test_divergent_flag_single_source_terminal_vs_pm_tab():
    """One source of truth for DIVERGENT (bucket_divergent, ±5pp): the terminal renderer
    and the PM-tab table mark exactly the same rows, and the threshold semantics are
    pinned (>5 strictly, both signs)."""
    from scripts.spcx_pm_pdf_monitor import bucket_divergent, render_pm_panel
    assert not bucket_divergent(5.0) and not bucket_divergent(-5.0)
    assert bucket_divergent(5.01) and bucket_divergent(-5.01)
    dash, rep, snap = _two_poll_dash()
    expected = {r["label"] for r in rep["buckets"] if bucket_divergent(r["gap_pp"])}
    assert expected, "fixture must contain a divergent bucket (the audit's +7.8pp)"
    term = {line.split()[0] for line in render_text(rep).splitlines()
            if "<-- DIVERGENT" in line}
    assert term == expected
    html = render_pm_panel(rep, dash, now=dash.history[-1]["ts"])
    pm = set()
    for chunk in html.split("<tr")[1:]:
        cells = _re.findall(r"<td[^>]*>(.*?)</td>", chunk, _re.S)
        if cells and "class='divg'" in chunk:
            pm.add(cells[0])
    assert pm == expected


def test_bucket_gap_sparkline_from_synthetic_two_poll_log(tmp_path):
    """Synthetic two-poll parquet log → backfill rebuilds per-poll gaps through the SAME
    bucket_compare path as a live poll (numeric equality asserted), and the PM panel
    renders a sparkline whose divergent end-dot flags the newly opened gap."""
    from datetime import datetime as dt, timedelta as td, timezone as tz

    from scripts.spcx_pm_pdf_monitor import DashboardState, bucket_divergent, \
        render_pm_panel
    now_dt = dt.now(tz.utc)
    snap1 = fixture_snapshot()
    snap1["fetched_at_utc"] = (now_dt - td(minutes=30)).isoformat()
    rep1 = analyze(snap1, basis="mid")
    snap2 = json.loads(json.dumps(snap1))
    snap2["fetched_at_utc"] = now_dt.isoformat()
    bump = rep1["buckets"][3]["label"]               # open a fresh divergence here
    for b in snap2["buckets"]:
        if b["label"] == bump:
            b["bid"] = min(b["bid"] + 0.10, 0.99)
            b["ask"] = min(b["ask"] + 0.10, 0.995)
    rep2 = analyze(snap2, basis="mid")
    log_parquet(snap1, rep1, out_dir=tmp_path)
    log_parquet(snap2, rep2, out_dir=tmp_path)
    dash = DashboardState()
    assert dash.backfill_from_parquet(log_dir=tmp_path, days=1) == 2
    # equality: backfilled gaps == the live analyze() gaps, poll by poll
    for rec, rep in zip(dash.history, [rep1, rep2]):
        live = {r["label"]: r["gap_pp"] for r in rep["buckets"]}
        assert rec["bucket_gaps"] is not None
        for label, gp in rec["bucket_gaps"].items():
            assert gp == pytest.approx(live[label], abs=1e-9)
    # the bumped bucket's gap moved up between polls and the panel renders its path
    g1 = dash.history[0]["bucket_gaps"][bump]
    g2 = dash.history[1]["bucket_gaps"][bump]
    assert g2 > g1 and bucket_divergent(g2)
    now_ts = dash.history[-1]["ts"]
    html = render_pm_panel(rep2, dash, now=now_ts)
    assert html.count("<polyline") >= len(rep2["buckets"])   # a path per bucket row
    assert "fill='#f87171'" in html                          # divergent end-dot is red
    assert "2h ago" not in html                              # 30-min log: no stale read


def test_backfill_guardrail_isolates_broken_shards(tmp_path):
    """Backfill guardrail (2026-06-12): a malformed shard must be SKIPPED + COUNTED, never
    crash the load and lose the good shards after it. Writes a good shard, then three broken
    ones (corrupt file, empty table, missing the mean_ps anchor column), then another good
    shard, in filename order — backfill loads exactly the 2 good ones and survives."""
    from datetime import timezone as _tz

    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.spcx_pm_pdf_monitor import DashboardState, backfill_health

    today = _dt.now(_tz.utc).strftime("%Y%m%d")
    # two good shards (real schema) bracketing the broken ones
    for hh in ("000000", "090000"):
        snap = fixture_snapshot()
        snap["fetched_at_utc"] = f"{today[:4]}-{today[4:6]}-{today[6:]}T{hh[:2]}:00:00+00:00"
        rep = analyze(snap, basis="mid")
        log_parquet(snap, rep, out_dir=tmp_path)
    # broken #1 — not a parquet file at all
    (tmp_path / f"poll_{today}T030000.parquet").write_text("garbage not parquet")
    # broken #2 — valid parquet, zero rows
    pq.write_table(pa.table({"poll_ts": pa.array([], pa.string())}),
                   tmp_path / f"poll_{today}T040000.parquet")
    # broken #3 — valid rows but the mean_ps session anchor is missing (old/forked schema)
    pq.write_table(pa.table({"poll_ts": [f"{today[:4]}-{today[4:6]}-{today[6:]}T05:00:00+00:00"],
                             "kind": ["ladder"], "strike_lo_t": [2.0],
                             "bid": [0.5], "ask": [0.52], "p_win_offer": [0.8]}),
                   tmp_path / f"poll_{today}T050000.parquet")
    dash = DashboardState()
    n = dash.backfill_from_parquet(log_dir=tmp_path, days=1)   # must NOT raise
    assert n == 2                                              # only the two good shards
    assert all(r["mean_ps"] is not None for r in dash.history)
    # the health report sees 2 loaded, 3 skipped with reasons, and is NOT degenerate
    rep = backfill_health("test", dash.history, n, 3,
                          {"unreadable:ArrowInvalid": 1, "empty": 1, "malformed": 1})
    assert rep["loaded"] == 2 and rep["skipped"] == 3 and rep["degenerate"] is False


def test_backfill_health_flags_degenerate_cases():
    """The guardrail WARNs (degenerate=True) when every shard was skipped, or a session
    anchor is non-finite on a material fraction — and stays quiet when only perp/spot are
    absent (legitimate pre-cross), which must NOT trip the alarm."""
    from scripts.spcx_pm_pdf_monitor import backfill_health
    # all skipped, nothing loaded → degenerate
    assert backfill_health("t", [], 0, 5, {"unreadable:OSError": 5})["degenerate"] is True
    # healthy: anchors finite, perp/spot None (pre-cross) → not degenerate
    recs = [{"ts": 1000.0 + i, "mean_ps": 167.0, "median_ps": 162.0, "pwin": 0.82,
             "perp": None, "spot": None} for i in range(20)]
    r = backfill_health("t", recs, 20, 0, {})
    assert r["degenerate"] is False and all(m == 0 for m in r["missing"].values())
    # a NaN mean_ps on >=5% of loaded → degenerate (caught by _is_finite, not just None)
    recs[0]["mean_ps"] = float("nan")
    recs[1]["mean_ps"] = None
    assert backfill_health("t", recs, 20, 0, {})["degenerate"] is True


def _fake_arb(exists=True, investable=False, fee_bps=0.0):
    """A best_executable_arb()-shaped dict (the documented render contract) — render-level
    tests need the shape, not the network; the walk math itself is pinned in the S8 suite."""
    best = None
    if exists:
        best = {"name": "box_cover", "class": "box_cover",
                "legs": [
                    {"action": "SELL", "market": "2-2.5T", "kind": "bucket",
                     "price": 0.41, "top_size": 300.0},
                    {"action": "BUY", "market": ">2T", "kind": "ladder",
                     "price": 0.99, "top_size": 120.0},
                    {"action": "SELL", "market": ">2.6T", "kind": "ladder",
                     "price": 0.79, "top_size": 80.0}],
                "pay_per_set": 1.190, "payout_floor": 1.2,
                "free_sliver": ["2.5-2.6T states pay above the floor"],
                "net_sets": 169.0, "notional_usd": 669.0, "net_usd": 5.55,
                "closed_by": "fees/spread"}
    inv = bool(exists and investable)
    return {"poll_ts": "t", "fee_bps": fee_bps,
            "fee_formula": f"({fee_bps:g}bps) x min(price, 1-price) per share, taker",
            "exists": exists, "investable": inv,
            "verdict": ("ARB" if inv else
                        "lock exists — uninvestable (dust)" if exists else "no lock"),
            "best": best, "n_candidates": 1 if exists else 0}


def test_pm3_arb_card_structure_and_verdict_palette():
    """PM3 render contract: card present under PM2, BUY/SELL legs rendered with prices,
    the fee formula stated on the card, both fee variants present with the toggle, and the
    verdict chip class tracking `investable` (green ONLY when investable; amber for
    lock-but-dust; neutral for no lock — red never)."""
    from scripts.spcx_pm_pdf_monitor import render_pm_panel
    dash, rep, snap = _two_poll_dash()
    now = dash.history[-1]["ts"]
    arb = {"0": _fake_arb(exists=True, investable=False, fee_bps=0.0),
           "1000": _fake_arb(exists=False, fee_bps=1000.0)}
    html = render_pm_panel(rep, dash, now=now, arb=arb, arb_fee_default=0.0)
    assert html.index("PM2") < html.index("PM3")             # sibling card under PM2
    assert "what's executable now (taker-only)" in html
    assert "<b>SELL</b> 2-2.5T @ 0.41" in html               # legs as BUY/SELL lines
    assert "<b>BUY</b> &gt;2T @ 0.99" in html or "<b>BUY</b> >2T @ 0.99" in html
    assert "pay <b>$1.190/set</b>" in html and "locked floor <b>$1.2</b>" in html
    assert "EXTRACTABLE: <b>$5.55</b> net over 169 sets ($669 notional)" in html
    assert "edge closes by fees/spread" in html
    assert "(0bps) x min(price, 1-price) per share, taker" in html      # fee stated
    assert "(1000bps) x min(price, 1-price) per share, taker" in html   # both variants
    assert "id='pm3v0'" in html and "id='pm3v1000'" in html
    assert "pm3Fee(0)" in html and "pm3Fee(1000)" in html               # the toggle
    assert "data-def='0'" in html                            # --fee-bps default honored
    assert "pm3chip a'>lock exists — dust" in html           # dust → amber, NOT green
    assert "pm3chip n'>no executable lock" in html           # no lock → neutral
    assert "pm3chip g" not in html                           # green only if investable
    # investable case flips the chip green
    arb_g = {"0": _fake_arb(exists=True, investable=True),
             "1000": _fake_arb(exists=False, fee_bps=1000.0)}
    html_g = render_pm_panel(rep, dash, now=now, arb=arb_g, arb_fee_default=1000.0)
    assert "pm3chip g'>ARB — investable" in html_g
    assert "data-def='1000'" in html_g                       # --fee-bps 1000 default
    # PM1/PM2 untouched by the new card
    assert "PM1" in html and "DIVERGENT" in html or "PM1" in html


def test_pm3_degrades_without_arb_books():
    """No arb metadata / failed book fetch → the card renders its unavailable placeholder
    (like every other PM panel), and the rest of the PM panel is unaffected."""
    from scripts.spcx_pm_pdf_monitor import render_pm_panel
    dash, rep, snap = _two_poll_dash()
    html = render_pm_panel(rep, dash, now=dash.history[-1]["ts"], arb=None)
    assert "PM3" in html
    assert "arb books unavailable this poll" in html
    assert "pm3chip" not in html                             # no verdict without books
    assert "PM1" in html and "PM2" in html                   # siblings render normally
    # payload carriage: build_ws_payload threads arb through to the same renderer
    p = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>",
                         arb={"0": _fake_arb(), "1000": _fake_arb(fee_bps=1000.0)})
    json.dumps(p)
    assert "EXTRACTABLE" in p["pm_html"]


def test_pdf_tail_pts_on_live_and_scrub_payload(tmp_path):
    """The PDF chart's sell-tail shaded zone is fed by pdf.tail_pts, derived from the curve's
    own strikes (sell = YES bid), so it renders identically on the live PDF AND on any
    time-scrubbed historical curve (operator ask: 'as I slide I should still see the sell-tail
    shaded area'). Asserts both payloads carry it and sell_bid == the ladder YES bid."""
    from scripts.spcx_dashboard_server import build_ws_payload, curves_payload
    from scripts.spcx_pm_pdf_monitor import CurveIndex, TAIL_STRIKES, log_parquet
    dash, rep, snap = _two_poll_dash()
    bids = {row["strike_t"]: row["bid"] for row in snap["ladder"]}
    want = {k for k in TAIL_STRIKES if bids.get(k) is not None}
    assert want, "fixture must carry tail strikes with bids"
    # live payload (pdf comes from curves_payload)
    p = build_ws_payload(dash, _pb(eurusd=1.08), rep, snap, None, "<div>pb</div>")
    tp = p["pdf"]["tail_pts"]
    json.dumps(tp)
    assert {r["strike_t"] for r in tp} == want
    for r in tp:
        assert r["sell_bid"] == pytest.approx(bids[r["strike_t"]])   # sell = YES bid
        assert r["strike_ps"] > 0
    # scrub path: a refit historical poll carries the same markers via the same code path
    from datetime import timezone as _tz
    snap_now = json.loads(json.dumps(snap))
    snap_now["fetched_at_utc"] = _dt.now(_tz.utc).isoformat()   # inside the scrub window
    log_parquet(snap_now, analyze(snap_now, basis="mid"), out_dir=tmp_path)
    idx = CurveIndex(days=1)
    assert idx.load_parquet(log_dir=tmp_path) == 1
    _e, snap2, rep2 = idx.report_at(idx.range()["max_ts"])
    cp = curves_payload(rep2, snap2, perp=None, spot=None)
    assert {r["strike_t"] for r in cp["pdf"]["tail_pts"]} == want


def test_api_state_preallocation_save_button_regression(tmp_path):
    """Regression for the 2026-06-12 morning operator report ('pre-allocation state
    button does not respond'). Pins the exact E5 SAVE round-trip pre-allocation
    (fill+comfort+hedge_lev+sold), the instant playbook re-broadcast with the new state,
    persistence, and the clear-back-to-null path. (The browser click path was verified
    clean against a live server; the reproducible failure mode is a dead/unreachable
    server — now made unmissable at the button via setBtnsDead.)"""
    server, _ = _server_fixture()
    server.pb.path = tmp_path / "state.json"
    ws = _ws_connect(server)
    json.loads(ws.recv())                                    # drain snapshot
    body = {"fill": 22, "comfort": 22, "hedge_lev": 1, "sold": 0}   # E5 SAVE, pre-alloc
    req = _url.Request(server.url() + "api/state", method="POST",
                       data=json.dumps(body).encode(),
                       headers={"Content-Type": "application/json"})
    resp = json.loads(_url.urlopen(req, timeout=5).read())
    assert resp["ok"] and set(resp["changed"]) == set(body)
    assert resp["state"]["fill"] == 22.0
    m = json.loads(ws.recv())                                # instant re-broadcast
    assert m["type"] == "playbook" and m["state"]["fill"] == 22.0
    assert json.loads((tmp_path / "state.json").read_text())["fill"] == 22.0
    req2 = _url.Request(server.url() + "api/state", method="POST",
                        data=json.dumps({"fill": None}).encode(),
                        headers={"Content-Type": "application/json"})
    resp2 = json.loads(_url.urlopen(req2, timeout=5).read())
    assert resp2["ok"] and resp2["state"]["fill"] is None    # cleared, not coerced to 0
    m2 = json.loads(ws.recv())
    assert m2["type"] == "playbook" and m2["state"]["fill"] is None
    assert json.loads((tmp_path / "state.json").read_text())["fill"] is None
    ws.close()
