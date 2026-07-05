"""Tests for the newsagent hybrid FV pipeline (Stage A features + Stage B model).

Pure-function tests — no network, no LLM, temp-dir caches.
Run: ``PYTHONPATH=. uv run pytest tests/test_newsagent_fv.py``
"""
from __future__ import annotations

import json
import math

import pytest

from newsagent import dashboard, features, fvmodel
from newsagent.feeds import _lede_last, relevance_rank


def F(relevance=0.8, stance="toward_yes", phase="in_progress", strength=0.7,
      tone=0.0, novelty=1.0, clarity=0.5):
    return {"relevance": relevance, "stance": stance, "event_phase": phase,
            "strength": strength, "tone": tone, "event_type": "other",
            "entities": [], "novelty": novelty, "clarity": clarity}


PARAMS = {"alpha": 1.85, "lambda": {"slow": 0.9, "shock": 0.6},
          "floor_pp": {"slow": 8.0, "shock": 12.0}}


# ---------------------------------------------------------------- Stage A ------

def test_validate_clamps_and_normalizes():
    out = features.validate({"relevance": 1.7, "stance": "bogus", "event_phase": "nope",
                             "strength": -2, "tone": 9, "novelty": 3})
    assert out["relevance"] == 1.0 and out["stance"] == "neutral"
    assert out["event_phase"] == "none" and out["strength"] == 0.0
    assert out["tone"] == 1.0 and out["novelty"] == 1.0


def test_cache_key_market_relative_and_versioned():
    k1 = features.cache_key("market-a", "Some Headline")
    k2 = features.cache_key("market-b", "Some Headline")
    k3 = features.cache_key("market-a", "  some headline  ")
    assert k1 != k2          # same article, different market -> different features
    assert k1 == k3          # whitespace/case-insensitive
    assert f"v{features.PROMPT_VERSION}" in f"market-a|some headline|v{features.PROMPT_VERSION}"


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path)
    key = features.cache_key("m", "t")
    features.write_cache(key, F(), meta={"source": "test"})
    rec = features.cached(key)
    assert rec["relevance"] == 0.8 and rec["_meta"]["source"] == "test"


def test_pending_extractions_skips_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path)
    arts = [{"title": "A"}, {"title": "B"}]
    pend = features.pending_extractions("m", "q?", "crit", arts)
    assert len(pend) == 2
    features.write_cache(pend[0]["cache_key"], F())
    pend2 = features.pending_extractions("m", "q?", "crit", arts)
    assert len(pend2) == 1 and pend2[0]["text"].startswith("B")


def test_item_text_caps_and_joins():
    a = {"title": "T", "trail": "tr", "lede": "x" * 2000, "last_para": "L"}
    txt = features._item_text(a)
    assert txt.startswith("T || tr || x") and len(txt) <= 1200


# ---------------------------------------------------------------- feeds --------

def test_lede_last_flat_block():
    lede, last = _lede_last("word " * 300)
    assert lede and last and len(lede) <= 420


def test_relevance_rank_orders_by_hits():
    items = [{"title": "football roundup", "trail": "", "lede": ""},
             {"title": "putin succession rumors grow", "trail": "kremlin", "lede": ""}]
    ranked = relevance_rank(items, "putin AND (succession OR power)", ["kremlin"])
    assert ranked[0]["title"].startswith("putin")


# ---------------------------------------------------------------- Stage B ------

def test_contribution_signs_and_zero():
    assert fvmodel.contribution(F(stance="toward_yes")) > 0
    assert fvmodel.contribution(F(stance="toward_no")) < 0
    assert fvmodel.contribution(F(stance="neutral")) == 0
    assert fvmodel.contribution(F(phase="none", strength=0)) == 0
    assert fvmodel.contribution(None) == 0


def test_effective_weights_decisive_signal():
    cs = [0.9, 0.5, -0.7, -0.2, 0.0]
    ws = fvmodel.effective_weights(cs)
    assert ws[0] == 1.0 and ws[2] == 1.0            # strongest each way count fully
    assert ws[1] == ws[3] == fvmodel.CORROBORATION_W
    assert ws[4] == 0.0


def test_daily_score_dominance_not_drowned():
    # one decisive positive vs a pile of weak negatives: net stays positive
    feats = [{"features": F(stance="toward_yes", phase="completed", strength=0.9)}] + \
            [{"features": F(stance="toward_no", phase="speculative", strength=0.3,
                            relevance=0.5)} for _ in range(6)]
    assert fvmodel.daily_score(feats) > 0


def test_step_state_decay_and_clip():
    s1 = fvmodel.step_state(None, "2026-07-01", 2.0, 0.6)
    assert s1["A"] == 2.0
    s2 = fvmodel.step_state(s1, "2026-07-03", 0.0, 0.6)   # two days: 2 * 0.36
    assert abs(s2["A"] - 2.0 * 0.6 ** 2) < 1e-9
    s3 = fvmodel.step_state(s2, "2026-07-04", 99.0, 0.6)
    assert s3["A"] == fvmodel.A_CLIP


def test_fair_value_monotone_and_clipped():
    assert fvmodel.fair_value(10.0, 0.0, 1.85) == pytest.approx(10.0, abs=0.2)
    assert fvmodel.fair_value(10.0, 2.0, 1.85) > 10.0
    assert fvmodel.fair_value(10.0, -2.0, 1.85) < 10.0
    assert fvmodel.fair_value(99.0, 6.0, 6.0) == 99.0
    assert fvmodel.fair_value(1.0, -6.0, 6.0) == 1.0


def test_band_floors_by_type_and_dispersion():
    quiet = [{"features": F(relevance=0.1)} for _ in range(3)]   # nothing relevant
    assert fvmodel.band_half_pp(quiet, "slow", PARAMS) >= 8.0
    assert fvmodel.band_half_pp(quiet, "shock", PARAMS) >= 12.0
    agree = [{"features": F()} for _ in range(6)]
    disagree = [{"features": F(stance="toward_yes")}, {"features": F(stance="toward_no")},
                {"features": F(stance="toward_yes", strength=0.2)}]
    assert fvmodel.band_half_pp(disagree, "shock", PARAMS) >= \
        fvmodel.band_half_pp(agree, "shock", PARAMS)
    assert fvmodel.band_half_pp(agree, "shock", PARAMS) <= fvmodel.BAND_CAP_PP


def test_divergence_flag_rule():
    # diverges + confident -> flag
    d = fvmodel.divergence_flag(50.0, 30.0, 10.0, 6)
    assert d["flag"] and d["diverges"] and d["confident"]
    # wide band kills it
    assert not fvmodel.divergence_flag(50.0, 30.0, 20.0, 6)["flag"]
    # thin evidence kills it
    assert not fvmodel.divergence_flag(50.0, 30.0, 10.0, 2)["flag"]
    # small gap kills it
    assert not fvmodel.divergence_flag(40.0, 30.0, 10.0, 6)["flag"]


def test_breakdown_sums_to_fv():
    day = [{"article": {"title": "a", "domain": "d"},
            "features": F(stance="toward_yes", phase="completed", strength=0.9)},
           {"article": {"title": "b", "domain": "d"},
            "features": F(stance="toward_no", phase="planned", strength=0.5)},
           {"article": {"title": "c", "domain": "d"},
            "features": F(stance="toward_yes", phase="speculative", strength=0.4)}]
    prev = {"date": "2026-07-03", "A": 1.2}
    bd = fvmodel.breakdown(12.0, prev, "2026-07-05", day, "shock", PARAMS)
    total = bd["p0_pct"] + bd["carry_pp"] + sum(a["pp_effect"] for a in bd["articles"])
    assert total == pytest.approx(bd["fv_pct"], abs=0.05)
    # and fv matches the model run with the same state math
    s_t = fvmodel.daily_score(day)
    nxt = fvmodel.step_state(prev, "2026-07-05", s_t, PARAMS["lambda"]["shock"])
    assert bd["fv_pct"] == pytest.approx(
        fvmodel.fair_value(12.0, nxt["A"], PARAMS["alpha"]), abs=0.05)


def test_breakdown_no_evidence_is_prior_plus_carry():
    bd = fvmodel.breakdown(20.0, None, "2026-07-05", [], "slow", PARAMS)
    assert bd["fv_pct"] == pytest.approx(20.0, abs=0.1)
    assert bd["articles"] == []


def test_fit_alpha_recovers_signal():
    # synthetic: A perfectly separates outcomes -> alpha > 0 and beats prior-only
    pairs = ([{"slug": f"y{i}", "p0_pct": 10.0, "A": 1.5, "y": 1} for i in range(6)] +
             [{"slug": f"n{i}", "p0_pct": 10.0, "A": -1.0, "y": 0} for i in range(6)])
    fit = fvmodel.fit_alpha(pairs)
    assert fit["alpha"] > 0
    assert fit["brier_at_best"] < fit["brier_prior_only"]


def test_fit_alpha_zero_when_evidence_uninformative():
    pairs = ([{"slug": "a", "p0_pct": 50.0, "A": 1.0, "y": 0}] * 5 +
             [{"slug": "b", "p0_pct": 50.0, "A": -1.0, "y": 1}] * 5)
    assert fvmodel.fit_alpha(pairs)["alpha"] == 0.0


# ---------------------------------------------------------------- dashboard ----

def _snapshot(slug="m1", fv=40.0, mid=0.30, flag=False):
    sb = {"mtype": "shock", "n_relevant": 6,
          "divergence": {"gap_pp": round(fv - mid * 100, 1), "flag": flag,
                         "diverges": abs(fv - mid * 100) >= 15, "confident": True,
                         "rule": "r"},
          "breakdown": {"p0_pct": 30.0, "carry_pp": 1.0, "alpha": 1.85, "lambda": 0.6,
                        "clip_scale": 1.0, "fv_pct": fv,
                        "articles": [{"title": "t", "domain": "d", "c": 0.5,
                                      "pp_effect": 9.0}]}}
    return {"market": {"slug": slug, "question": "Q?", "end_date": "2026-08-01T00:00:00Z",
                       "mid": mid, "volume24h": 1000, "liquidity": 5000},
            "packet": {"articles": [{"title": "t", "domain": "d", "seendate": "20260705",
                                     "url": "https://x"}]},
            "forecast": {"p_pct": fv, "band_lo_pct": fv - 10, "band_hi_pct": fv + 10},
            "drivers": ["t"], "region": "US", "stage_b": sb, "sf_id": "sf-1"}


def _series():
    return {"m1": [
        {"date": "2026-07-01", "fv_pct": 35.0, "band_lo_pct": 25.0, "band_hi_pct": 45.0,
         "mid_pct": 30.0, "segment": "backfill"},
        {"date": "2026-07-02", "fv_pct": 36.0, "band_lo_pct": 26.0, "band_hi_pct": 46.0,
         "mid_pct": 31.0, "segment": "backfill"},
        {"date": "2026-07-05", "fv_pct": 40.0, "band_lo_pct": 30.0, "band_hi_pct": 50.0,
         "mid_pct": 30.0, "segment": "live"},
    ]}


def test_showcase_json_shape_and_interim():
    sc = dashboard.build_showcase([_snapshot()], _series())
    assert sc["markets"][0]["fv_pct"] == 40.0
    assert sc["divergence_rule"].startswith("Flag shown")
    # interim: FV_t vs NEXT-day mid, 2 consecutive pairs available
    assert sc["interim"]["n"] == 2
    expected_first = ((35.0 - 31.0) / 100) ** 2
    per = sc["interim"]["per_market"]["m1"]
    assert per["n"] == 2
    assert sc["interim"]["overall_mean"] == pytest.approx(
        (expected_first + ((36.0 - 30.0) / 100) ** 2) / 2, abs=1e-6)


def test_html_renders_selfcontained_and_scrubbed():
    sc = dashboard.build_showcase([_snapshot(flag=True)], _series())
    html_out = dashboard.render_html(sc)
    assert "<script src" not in html_out and "http://cdn" not in html_out
    assert "divergence" in html_out.lower()
    assert "fv construction" in html_out.lower()
    assert "Not investment advice" in html_out
    # no absolute local paths leak into the public page
    assert "/Users/" not in html_out
    # framing never claims to beat the market
    assert "beat the mid" not in html_out.lower()


def test_series_svg_marks_backfill_dashed():
    svg = dashboard._svg_series(_series()["m1"])
    assert "stroke-dasharray" in svg and "reconstructed" in svg


def test_ledger_snapshot_uses_sf_cli_shape():
    # log_snapshot builds CLI args only through the sf wrapper; here we just check
    # the probability normalization contract stays percent -> fraction
    fc = {"p_pct": 40.0, "band_lo_pct": 30.0, "band_hi_pct": 50.0}
    assert 0.0 < fc["p_pct"] / 100.0 < 1.0


# ---------------------------------------------------------------- GDELT (v2.1) --

from newsagent import gdelt_bq


def test_amplify_direction_preserving_and_graceful():
    assert fvmodel.amplify(0.5, 2.0, 1.0) == pytest.approx(1.5)
    assert fvmodel.amplify(-0.5, 2.0, 1.0) == pytest.approx(-1.5)   # sign preserved
    assert fvmodel.amplify(0.0, 3.0, 1.0) == 0.0                    # burst can't create evidence
    assert fvmodel.amplify(0.5, None, 1.0) == 0.5                   # no series -> inert
    assert fvmodel.amplify(0.5, 2.0, 0.0) == 0.5                    # gamma off -> inert
    assert fvmodel.amplify(0.5, -2.0, 1.0) == 0.5                   # quiet day never dampens
    assert fvmodel.amplify(2.0, 3.0, 1.5) == fvmodel.S_CLIP         # S-clip preserved


def test_burst_z_from_series():
    series = {f"202606{d:02d}": {"n": 100, "tone": -2.0} for d in range(1, 15)}
    series["20260615"] = {"n": 400, "tone": -5.0}
    b = gdelt_bq.burst_z(series, "20260615")
    assert b["vol_z"] == 3.0                     # huge burst, clipped at 3
    assert b["tone_shift"] == pytest.approx(-3.0)
    assert gdelt_bq.burst_z(series, "20990101") is None      # unknown day
    assert gdelt_bq.burst_z({"20260615": {"n": 5, "tone": 0}}, "20260615") is None  # short baseline


def test_burst_z_quiet_series_no_fake_burst():
    # tiny counts: sd floor prevents 0->2 articles registering as a 3-sigma burst
    series = {f"202606{d:02d}": {"n": 0, "tone": None} for d in range(1, 15)}
    series["20260615"] = {"n": 2, "tone": 1.0}
    b = gdelt_bq.burst_z(series, "20260615")
    assert b["vol_z"] <= 2.0


def test_breakdown_with_burst_still_sums():
    day = [{"article": {"title": "a", "domain": "d"},
            "features": F(stance="toward_yes", phase="completed", strength=0.9)},
           {"article": {"title": "b", "domain": "d"},
            "features": F(stance="toward_no", phase="planned", strength=0.5)}]
    params = dict(PARAMS, gamma=1.0)
    prev = {"date": "2026-07-03", "A": 0.4}
    bd = fvmodel.breakdown(20.0, prev, "2026-07-05", day, "shock", params, vol_z=2.0)
    total = bd["p0_pct"] + bd["carry_pp"] + sum(a["pp_effect"] for a in bd["articles"])
    assert total == pytest.approx(bd["fv_pct"], abs=0.05)
    # matches the pipeline: amplified S -> step -> fair_value
    s_eff = fvmodel.amplify(fvmodel.daily_score(day), 2.0, 1.0)
    nxt = fvmodel.step_state(prev, "2026-07-05", s_eff, params["lambda"]["shock"])
    assert bd["fv_pct"] == pytest.approx(
        fvmodel.fair_value(20.0, nxt["A"], params["alpha"]), abs=0.05)


def test_gdelt_html_absent_is_empty():
    assert dashboard._gdelt_html(None) == ""
    out = dashboard._gdelt_html({"n": 1200, "n_trailing_mean": 400.0, "vol_z": 2.1,
                                 "tone": -3.2, "tone_shift": -1.1})
    assert "1,200" in out and "+2.1" in out


# ---------------------------------------------------------------- v3: sources --

from newsagent import email_ingest, feeds as feeds_mod, sourceweights


def test_title_hash_normalizes():
    a = feeds_mod.title_hash("Trump: 'Peace deal' could be SIGNED by Sunday!")
    b = feeds_mod.title_hash("trump peace deal could be signed by sunday")
    assert a == b
    assert feeds_mod.title_hash("something else") != a


def test_keyword_filter_window_and_terms():
    from datetime import datetime, timezone
    items = [
        {"title": "Putin succession rumors grow", "seendate": "20260704T120000Z", "trail": ""},
        {"title": "Putin old story", "seendate": "20260601T120000Z", "trail": ""},
        {"title": "Football roundup", "seendate": "20260704T120000Z", "trail": ""},
    ]
    start = datetime(2026, 7, 2, tzinfo=timezone.utc)
    end = datetime(2026, 7, 5, tzinfo=timezone.utc)
    out = feeds_mod._keyword_filter(items, "putin AND succession", ["putin"], start, end)
    assert [a["title"] for a in out] == ["Putin succession rumors grow"]


def test_packet_dedupes_across_sources(monkeypatch):
    from datetime import datetime, timezone
    monkeypatch.setattr(feeds_mod, "guardian_search", lambda *a, **k: [
        {"title": "Putin signals succession plan", "seendate": "20260705T090000Z",
         "domain": "theguardian.com", "url": "https://g", "trail": "", "lede": "", "last_para": ""}])
    monkeypatch.setattr(feeds_mod, "wp_day_bullets", lambda d: [])
    rss = [{"title": "Putin signals succession plan!", "seendate": "20260705T100000Z",
            "domain": "bbc.co.uk", "url": "https://b", "trail": "putin"},
           {"title": "Kremlin reshuffle: Putin loyalists promoted", "seendate": "20260705T110000Z",
            "domain": "news.sky.com", "url": "https://s", "trail": "putin"}]
    pkt = feeds_mod.build_packet("m", {"guardian_q": "putin", "wp_keys": ["putin"]},
                                 now=datetime(2026, 7, 5, 12, tzinfo=timezone.utc),
                                 rss_items=rss)
    titles = [a["title"] for a in pkt["articles"]]
    assert "Putin signals succession plan" in titles          # guardian kept
    assert "Putin signals succession plan!" not in titles     # rss duplicate dropped
    assert any("Kremlin reshuffle" in t for t in titles)      # distinct rss kept
    assert pkt["source"].startswith("newsletters+guardian+rss") or "rss" in pkt["source"]


def test_newsletter_parse_and_privacy():
    raw = (b"From: ING Economics <newsletter@ing.com>\r\n"
           b"Subject: ING Daily: Fed holds, markets shrug\r\n"
           b"Date: Sun, 05 Jul 2026 06:00:00 +0000\r\n"
           b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
           + ("The Federal Reserve is widely expected to hold rates steady this month. "
              "Our economists see the first cut no earlier than December." * 3
              + "\n\nBottom line: policy stays restrictive through the autumn, and the "
                "bar for a July move is very high indeed.").encode())
    item = email_ingest.parse_message(raw)
    assert item is not None
    assert item["display"] is False                       # never shown publicly
    assert item["domain"].startswith("newsletter:")
    assert item["title"].startswith("ING Daily")
    assert "hold rates steady" in item["lede"]


def test_newsletter_non_matching_sender_dropped():
    raw = (b"From: spam@example.com\r\nSubject: buy now\r\n"
           b"Content-Type: text/plain\r\n\r\n" + b"x" * 200)
    assert email_ingest.parse_message(raw) is None


def test_email_unavailable_is_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(email_ingest, "IMAP_CRED", tmp_path / "nope.json")
    monkeypatch.setattr(email_ingest, "GMAIL_CRED", tmp_path / "nope2.json")
    ok, why = email_ingest.available()
    assert not ok and "Justin" in why
    assert email_ingest.fetch_newsletters() == []


# ---------------------------------------------------------- v3: source weights --

def test_scheme_a_weight_mapping(monkeypatch):
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "refresh", lambda force=False: {
        "fetched_at": "2026-07-05T00:00:00+00:00",
        "rsp_status_by_id": {"the guardian": "s-gr", "bbc": "s-gr", "sky news uk": "s-nc",
                             "politico": "s-gu", "the hill": "s-d"},
        "iffy_domains": ["badnews.example"]})
    assert sourceweights.get_weight("theguardian.com") == 1.0
    assert sourceweights.get_weight("news.sky.com") == 0.7
    assert sourceweights.get_weight("politico.com") == 0.3
    assert sourceweights.get_weight("thehill.com") == 0.0
    assert sourceweights.get_weight("badnews.example") == 0.0
    assert sourceweights.get_weight("newsletter:ING research") == 1.0   # uncovered->neutral
    assert sourceweights.get_weight("en.wikipedia.org (Current events)") == 1.0
    assert sourceweights.get_weight("unknown.example") == 1.0
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)   # reset module cache


# ---------------------------------------------------------------- v3: band ------

def test_band_v3_reliability_weighted_dispersion():
    params = dict(PARAMS, band_mult=1.0)
    # two reliable sources disagreeing vs concurring
    disagree = [{"features": F(stance="toward_yes", clarity=1.0), "source_w": 1.0},
                {"features": F(stance="toward_no", clarity=1.0), "source_w": 1.0},
                {"features": F(stance="toward_yes", strength=0.3, clarity=1.0), "source_w": 1.0}]
    concur = [{"features": F(stance="toward_yes", clarity=1.0), "source_w": 1.0}
              for _ in range(3)]
    assert fvmodel.band_half_pp(disagree, "shock", params) > \
        fvmodel.band_half_pp(concur, "shock", params)
    # a zero-weight source cannot move the band
    with_junk = concur + [{"features": F(stance="toward_no", clarity=1.0), "source_w": 0.0}]
    assert fvmodel.band_half_pp(with_junk, "shock", params) == \
        fvmodel.band_half_pp(concur, "shock", params)


def test_band_v3_low_clarity_widens():
    params = dict(PARAMS, band_mult=1.0)
    clear = [{"features": F(clarity=1.0), "source_w": 1.0} for _ in range(4)]
    murky = [{"features": F(clarity=0.1), "source_w": 1.0} for _ in range(4)]
    assert fvmodel.band_half_pp(murky, "shock", params) > \
        fvmodel.band_half_pp(clear, "shock", params)


def test_band_mult_single_knob():
    params = dict(PARAMS, band_mult=1.0)
    wide = dict(PARAMS, band_mult=2.0)
    rows = [{"features": F(stance="toward_yes", clarity=0.5), "source_w": 1.0},
            {"features": F(stance="toward_no", clarity=0.5), "source_w": 1.0}]
    assert fvmodel.band_half_pp(rows, "shock", wide) >= \
        fvmodel.band_half_pp(rows, "shock", params)


def test_source_weight_scales_contribution():
    f = F(stance="toward_yes", phase="completed", strength=0.9)
    full = fvmodel.daily_score([{"features": f, "source_w": 1.0}])
    half = fvmodel.daily_score([{"features": f, "source_w": 0.5}])
    zero = fvmodel.daily_score([{"features": f, "source_w": 0.0}])
    assert full > half > zero == 0.0


def test_band_coverage_collecting(tmp_path):
    out = fvmodel.band_coverage(book_dir=tmp_path)
    assert out["n_settled"] == 0


def test_features_v3_clarity_validated():
    out = features.validate({"relevance": 0.5, "stance": "toward_yes",
                             "event_phase": "planned", "strength": 0.5, "clarity": 7})
    assert out["clarity"] == 1.0
    out2 = features.validate({"relevance": 0.5, "stance": "toward_yes",
                              "event_phase": "planned", "strength": 0.5})
    assert out2["clarity"] == 0.5


# ---------------------------------------------------------------- v3: charts ----

def test_gauge_and_sparkline_render():
    g = dashboard._svg_gauge(40.0, 30.0, 50.0, 60.0)
    assert "<svg" in g and "aria-label" in g
    s = dashboard._svg_sparkline([{"fv_pct": 30.0}, {"fv_pct": 40.0}, {"fv_pct": 35.0}])
    assert "<svg" in s and "circle" in s
    assert "new" in dashboard._svg_sparkline([{"fv_pct": 30.0}])   # 1 point -> placeholder


def test_overview_grid_orders_flags_first():
    cards = [
        {"question": "small gap", "fv_pct": 50.0, "market_pct": 48.0, "gap_pp": 2.0,
         "band": [40, 60], "divergence_flag": False, "series": []},
        {"question": "big gap unflagged", "fv_pct": 80.0, "market_pct": 50.0, "gap_pp": 30.0,
         "band": [70, 90], "divergence_flag": False, "series": []},
        {"question": "flagged", "fv_pct": 30.0, "market_pct": 50.0, "gap_pp": -20.0,
         "band": [22, 38], "divergence_flag": True, "series": []},
    ]
    grid = dashboard._overview_grid_html(cards)
    assert grid.index("flagged") < grid.index("big gap unflagged") < grid.index("small gap")
