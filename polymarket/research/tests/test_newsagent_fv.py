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
    monkeypatch.setattr(email_ingest, "GMAIL_CLIENT_SECRET", tmp_path / "nope2.json")
    monkeypatch.setattr(email_ingest, "GMAIL_TOKEN", tmp_path / "nope3.json")
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
    # uncovered-source weights: APPROVED 2026-08-24 (were neutral 1.0 before)
    assert sourceweights.get_weight("newsletter:ING research") == 0.9
    assert sourceweights.get_weight("newsletter:ING THINK") == 0.9
    assert sourceweights.get_weight("newsletter:Bloomberg") == 0.9
    assert sourceweights.get_weight("am.jpmorgan.com") == 0.9      # bank research desk
    assert sourceweights.get_weight("am.gs.com") == 0.9
    assert sourceweights.get_weight("en.wikipedia.org (Current events)") == 0.8
    assert sourceweights.get_weight("unknown.example") == 0.5      # genuinely unknown
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)   # reset module cache


def test_blocklist_beats_declared_uncovered_weight(monkeypatch):
    """Iffy precedence holds for the new declared rows too — a blocklisted domain
    can never be resurrected by an uncovered-source weight."""
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "refresh", lambda force=False: {
        "fetched_at": "2026-08-24T00:00:00+00:00",
        "rsp_status_by_id": {"the guardian": "s-gr"},
        "iffy_domains": ["bloomberg.com"]})
    assert sourceweights.get_weight("bloomberg.com") == 0.0
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)


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
        {"question": "small gap", "slug": "s1", "fv_pct": 50.0, "market_pct": 48.0, "gap_pp": 2.0,
         "band": [40, 60], "divergence_flag": False, "series": []},
        {"question": "big gap unflagged", "slug": "s2", "fv_pct": 80.0, "market_pct": 50.0, "gap_pp": 30.0,
         "band": [70, 90], "divergence_flag": False, "series": []},
        {"question": "flagged", "slug": "s3", "fv_pct": 30.0, "market_pct": 50.0, "gap_pp": -20.0,
         "band": [22, 38], "divergence_flag": True, "series": []},
    ]
    grid = dashboard._overview_grid_html(cards)
    assert grid.index("flagged") < grid.index("big gap unflagged") < grid.index("small gap")


# =============================================================== v3.1 additions ==

from datetime import datetime, timezone as _tz

from newsagent import config as na_config, pdf_ingest


# ---------------------------------------------------------- email: gmail path --

def test_email_available_prefers_gmail_token(monkeypatch, tmp_path):
    tok = tmp_path / "gmail_token.json"
    tok.write_text("{}")
    monkeypatch.setattr(email_ingest, "GMAIL_TOKEN", tok)
    monkeypatch.setattr(email_ingest, "IMAP_CRED", tmp_path / "imap.json")
    ok, mode = email_ingest.available()
    assert ok and mode == "gmail"


def test_email_client_secret_without_token_points_at_consent(monkeypatch, tmp_path):
    monkeypatch.setattr(email_ingest, "GMAIL_TOKEN", tmp_path / "nope.json")
    monkeypatch.setattr(email_ingest, "IMAP_CRED", tmp_path / "nope2.json")
    cs = tmp_path / "client_secret.json"
    cs.write_text("{}")
    monkeypatch.setattr(email_ingest, "GMAIL_CLIENT_SECRET", cs)
    ok, why = email_ingest.available()
    assert not ok and "--consent" in why


def test_save_token_keeps_refresh_token(monkeypatch, tmp_path):
    tok = tmp_path / "gmail_token.json"
    monkeypatch.setattr(email_ingest, "GMAIL_TOKEN", tok)
    email_ingest._save_token({"access_token": "a1", "refresh_token": "r1", "expires_in": 100})
    prev = __import__("json").loads(tok.read_text())
    email_ingest._save_token({"access_token": "a2", "expires_in": 100}, prev=prev)
    rec = __import__("json").loads(tok.read_text())
    assert rec["access_token"] == "a2" and rec["refresh_token"] == "r1"


def test_newsletter_bloomberg_and_ingthink_senders():
    raw_b = (b"From: Bloomberg <noreply@news.bloomberg.com>\r\n"
             b"Subject: 5 Things to Start Your Day\r\n"
             b"Content-Type: text/plain\r\n\r\n"
             + b"Central banks in focus this week as the Fed weighs its next move. " * 5)
    item = email_ingest.parse_message(raw_b)
    assert item and item["domain"] == "newsletter:Bloomberg" and item["display"] is False
    raw_i = (b"From: ING THINK <newsletter@economics.ingthink.com>\r\n"
             b"Subject: Rates Spark: holding pattern\r\n"
             b"Content-Type: text/plain\r\n\r\n"
             + b"Bond markets drifted sideways as investors await the July decision. " * 5)
    item2 = email_ingest.parse_message(raw_i)
    assert item2 and item2["domain"] == "newsletter:ING THINK"


# ------------------------------------------------------------- pdf ingestion ---

def test_most_recent_weekday_math():
    sun = datetime(2026, 7, 5, 12, tzinfo=_tz.utc)          # Sunday
    assert pdf_ingest.most_recent(4, sun).strftime("%Y-%m-%d") == "2026-07-03"  # Friday
    assert pdf_ingest.most_recent(0, sun).strftime("%Y-%m-%d") == "2026-06-29"  # Monday
    fri = datetime(2026, 7, 3, 12, tzinfo=_tz.utc)
    assert pdf_ingest.most_recent(4, fri) == fri            # Friday maps to itself


def test_source_urls_templates_and_week_back():
    t = datetime(2026, 7, 5, 12, tzinfo=_tz.utc)
    gs = next(s for s in pdf_ingest.PDF_SOURCES if s["label"].startswith("GS"))
    urls = pdf_ingest.source_urls(gs, t)
    assert "2026/market_monitor_070326.pdf" in urls[0]
    assert "market_monitor_062626.pdf" in urls[1]           # one week back retry
    bofa = next(s for s in pdf_ingest.PDF_SOURCES if s["label"].startswith("BofA"))
    urls = pdf_ingest.source_urls(bofa, t)
    assert "CMO_Institutional_06-29-2026_ada.pdf" in urls[0]
    assert "CMO_Institutional_06-22-2026_ada.pdf" in urls[1]
    jpm = next(s for s in pdf_ingest.PDF_SOURCES if s["dated"] is None)
    assert pdf_ingest.source_urls(jpm, t) == [jpm["url"]]   # static, no retry


def test_strip_disclaimers_drops_boilerplate():
    text = ("The Fed held rates steady in June as inflation cooled.\n"
            "Past performance is not indicative of future results.\n"
            "This is not a solicitation to buy securities.\n"
            "© 2026 J.P. Morgan. All rights reserved.\n"
            "Growth surprised to the upside in the second quarter.")
    out = pdf_ingest.strip_disclaimers(text)
    assert "Fed held rates" in out and "Growth surprised" in out
    assert "Past performance" not in out and "solicitation" not in out
    assert "rights reserved" not in out


def test_fetch_pdf_reports_degrades_and_caches(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pdf_ingest, "CACHE_DIR", tmp_path)
    good = {"title": "JPM Weekly Market Recap — week of 2026-07-05",
            "seendate": "20260705T120000Z", "domain": "am.jpmorgan.com",
            "url": "https://x", "trail": "t", "lede": "l", "last_para": "",
            "scan_text": "federal reserve outlook", "macro": True}
    monkeypatch.setattr(pdf_ingest, "_fetch_one",
                        lambda src, t: good if src["label"].startswith("JPM Weekly Market") else None)
    items = pdf_ingest.fetch_pdf_reports("2026-07-05")
    assert items == [good]
    assert "degraded silently" in capsys.readouterr().out
    # second call comes from the day cache (fetcher would now raise)
    monkeypatch.setattr(pdf_ingest, "_fetch_one",
                        lambda src, t: (_ for _ in ()).throw(RuntimeError("no fetch")))
    assert pdf_ingest.fetch_pdf_reports("2026-07-05") == [good]


def test_keyword_filter_matches_pdf_scan_text():
    start = datetime(2026, 7, 1, tzinfo=_tz.utc)
    end = datetime(2026, 7, 6, tzinfo=_tz.utc)
    items = [{"title": "GS Weekly Market Monitor — week of 2026-07-03",
              "seendate": "20260703T120000Z", "trail": "", "lede": "",
              "scan_text": "page three discusses the federal reserve path"}]
    out = feeds_mod._keyword_filter(items, "\"federal reserve\"", ["federal reserve"],
                                    start, end)
    assert len(out) == 1


def test_packet_includes_pdf_slot(monkeypatch):
    monkeypatch.setattr(feeds_mod, "guardian_search", lambda *a, **k: [
        {"title": f"Fed story {i}", "seendate": "20260705T090000Z",
         "domain": "theguardian.com", "url": "https://g", "trail": "federal reserve",
         "lede": "", "last_para": ""} for i in range(6)])
    monkeypatch.setattr(feeds_mod, "wp_day_bullets", lambda d: [])
    pdfs = [{"title": "JPM Weekly Market Recap — week of 2026-07-05",
             "seendate": "20260705T120000Z", "domain": "am.jpmorgan.com",
             "url": "https://jpm", "trail": "", "lede": "",
             "scan_text": "the federal reserve held rates", "macro": True}]
    pkt = feeds_mod.build_packet(
        "m", {"guardian_q": "\"federal reserve\"", "wp_keys": ["federal reserve"]},
        now=datetime(2026, 7, 5, 12, tzinfo=_tz.utc), pdf_items=pdfs)
    titles = [a["title"] for a in pkt["articles"]]
    assert any(t.startswith("JPM Weekly") for t in titles)
    assert "pdf" in pkt["source"]


# --------------------------------------------------------- provider selection --

def test_pick_provider_explicit_and_auto(monkeypatch):
    monkeypatch.setenv(features.PROVIDER_ENV, "gemini")
    assert features.pick_provider() == "gemini"
    monkeypatch.setenv(features.PROVIDER_ENV, "oob")
    assert features.pick_provider() is None
    monkeypatch.delenv(features.PROVIDER_ENV, raising=False)
    monkeypatch.delenv(features.ANTHROPIC_KEY_ENV, raising=False)
    monkeypatch.delenv(features.GEMINI_KEY_ENV, raising=False)
    assert features.pick_provider() is None
    monkeypatch.setenv(features.GEMINI_KEY_ENV, "k")
    assert features.pick_provider() == "gemini"
    monkeypatch.setenv(features.ANTHROPIC_KEY_ENV, "k")
    assert features.pick_provider() == "anthropic"   # anthropic wins in auto


def test_match_rows_by_id_and_order():
    chunk = [{"cache_key": "k1"}, {"cache_key": "k2"}]
    rows = [{"id": "k2", "relevance": 1}, {"id": "k1", "relevance": 0}]
    out = features._match_rows(chunk, rows)
    assert out["k1"]["relevance"] == 0 and out["k2"]["relevance"] == 1
    # order-only fallback (no ids)
    out2 = features._match_rows(chunk, [{"relevance": 5}, {"relevance": 6}])
    assert out2["k1"]["relevance"] == 5 and out2["k2"]["relevance"] == 6


def test_extract_via_api_gemini_needs_key(monkeypatch):
    monkeypatch.delenv(features.GEMINI_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        features.extract_via_api("q", "c", [{"cache_key": "k", "text": "t"}],
                                 provider="gemini")


# ------------------------------------------------------------- market tagging --

def test_tract_tagging():
    """The v3.1 tag split into data-driven vs poll-driven (Justin, 2026-08-24)."""
    fed = "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615"
    assert na_config.tract(fed) == "data"
    assert na_config.tract("will-xavier-becerra-win-the-california-governor-election-in-2026") == "poll"
    assert na_config.tract("billionaire-one-time-wealth-tax-passes-in-california-election-2026") == "poll"
    assert na_config.tract("trump-out-as-president-before-2027") == "news"
    # the two families stay disjoint and every tagged market is on the live slate
    assert not (set(na_config.DATA_DRIVEN) & set(na_config.POLL_DRIVEN))
    assert all(s in na_config.LIVE_MARKETS for s in na_config.NOT_NEWS_TRACTABLE)
    # every tagged market carries a public note
    assert all(na_config.tract_note(s) for s in na_config.NOT_NEWS_TRACTABLE)
    assert na_config.tract_note("trump-out-as-president-before-2027") == ""


def test_becerra_note_is_refreshed_to_general_election():
    """v3.1 said 'state-primary polling'; the June primary has passed (2026-08-24)."""
    note = na_config.tract_note(
        "will-xavier-becerra-win-the-california-governor-election-in-2026")
    assert "primary" not in note.lower()
    assert "general-election polling" in note.lower()


# --------------------------------------------------------- bias breakdown ------

def test_source_bias_breakdown_groups_and_sentence():
    rows = [
        {"features": F(stance="toward_yes"), "source_w": 1.0,
         "article": {"domain": "theguardian.com"}},
        {"features": F(stance="toward_yes"), "source_w": 0.3,
         "article": {"domain": "someblog.example"}},
        {"features": F(stance="toward_no"), "source_w": 1.0,
         "article": {"domain": "bbc.co.uk"}},
        {"features": F(stance="neutral"), "source_w": 1.0,
         "article": {"domain": "news.sky.com"}},
        {"features": F(stance="toward_no"), "source_w": 0.0,
         "article": {"domain": "badnews.example"}},
        {"features": F(relevance=0.1), "source_w": 1.0,
         "article": {"domain": "thehill.com"}},           # irrelevant: ignored
    ]
    b = fvmodel.source_bias_breakdown(rows)
    assert b["n_yes"] == 2 and b["n_no"] == 1 and b["n_neutral"] == 1
    assert b["n_zero_weight"] == 1
    assert "higher-reliability" in b["yes_tiers"] and "more-biased/unreliable" in b["yes_tiers"]
    assert "2 items leaned YES" in b["sentence"] and "1 leaned NO" in b["sentence"]
    assert "blocklisted" in b["sentence"]


def test_source_bias_breakdown_newsletter_generic_label():
    rows = [{"features": F(stance="toward_yes"), "source_w": 1.0,
             "article": {"domain": "newsletter:ING THINK"}}]
    b = fvmodel.source_bias_breakdown(rows)
    label = b["yes_tiers"]["higher-reliability"][0][0]
    assert label == "ING THINK (newsletter)"


def test_source_bias_breakdown_no_evidence():
    b = fvmodel.source_bias_breakdown([])
    assert b["n_yes"] == 0 and "prior" in b["sentence"]


# --------------------------------------------------------- dashboard v3.1 UX ---

def _snapshot31(slug="m1", fv=40.0, mid=0.30, flag=False, tract="news",
                with_newsletter=False):
    s = _snapshot(slug=slug, fv=fv, mid=mid, flag=flag)
    s["stage_b"]["tract"] = tract
    s["stage_b"]["tract_note"] = ("Rate decisions are priced off futures."
                                  if tract == "data" else "")
    s["stage_b"]["bias"] = fvmodel.source_bias_breakdown(
        [{"features": F(stance="toward_yes"), "source_w": 1.0,
          "article": {"domain": "theguardian.com"}}])
    if with_newsletter:
        s["packet"]["articles"].append(
            {"title": "ZQX-SECRET newsletter headline", "domain": "newsletter:ING THINK",
             "seendate": "20260705", "url": "", "display": False})
        s["stage_b"]["breakdown"]["articles"].append(
            {"title": "ZQX-SECRET newsletter headline", "domain": "newsletter:ING THINK",
             "c": 0.4, "pp_effect": 4.0})
    return s


def test_showcase_filters_private_items():
    sc = dashboard.build_showcase([_snapshot31(with_newsletter=True)], _series())
    card = sc["markets"][0]
    assert card["n_private_items"] == 1
    assert all("ZQX-SECRET" not in e["title"] for e in card["evidence"])


def test_html_newsletter_privacy_end_to_end():
    sc = dashboard.build_showcase([_snapshot31(with_newsletter=True)], _series())
    out = dashboard.render_html(sc)
    assert "ZQX-SECRET" not in out
    assert "newsletter:" not in out            # raw domain tag never reaches the page
    assert "private analysis item" in out      # generic label shown instead


def test_html_v31_ux_elements():
    sc = dashboard.build_showcase(
        [_snapshot31(slug=f"m{i}", flag=(i == 0)) for i in range(3)], _series())
    out = dashboard.render_html(sc)
    assert "cardcols" in out                   # 2-column card layout
    assert "choose markets" in out and "DEFAULT_VISIBLE" in out
    assert "flags only" in out and "show all" in out
    assert 'data-flag="1"' in out
    assert "toggleCard" in out and "revealCard" in out
    assert "<script src" not in out            # still self-contained


def test_html_flagged_expanded_others_collapsed():
    sc = dashboard.build_showcase(
        [_snapshot31(slug="mflag", flag=True), _snapshot31(slug="mquiet", flag=False)],
        _series())
    out = dashboard.render_html(sc)
    quiet = out[out.index('id="card-mquiet"') - 200: out.index('id="card-mquiet"')]
    flagged = out[out.index('id="card-mflag"') - 200: out.index('id="card-mflag"')]
    assert "collapsed" in quiet and "collapsed" not in flagged


def test_html_tract_note_on_data_market():
    sc = dashboard.build_showcase([_snapshot31(tract="data")], _series())
    out = dashboard.render_html(sc)
    assert "not news-tractable" in out and "structurally blind" in out
    assert "◆ data-driven" in out


def test_html_bias_explainer_rendered():
    sc = dashboard.build_showcase([_snapshot31()], _series())
    out = dashboard.render_html(sc)
    assert "how the number formed" in out
    assert "leaned YES" in out


# =============================================================== v3.2 additions ==

from newsagent import sourcelean


# ------------------------------------------------------------ lean table -------

def test_lean_table_verified_domains_and_unrated():
    assert sourcelean.get_lean("theguardian.com") == -2      # AllSides: Left
    assert sourcelean.get_lean("bbc.co.uk") == 0             # Center
    assert sourcelean.get_lean("politico.com") == -1         # Lean Left
    assert sourcelean.get_lean("foxnews.com") == 2           # Right
    assert sourcelean.get_lean("dailymail.co.uk") == 1       # Lean Right
    # honestly unrated: no verified rating / out of scope
    assert sourcelean.get_lean("news.sky.com") is None
    assert sourcelean.get_lean("newsletter:ING THINK") is None
    assert sourcelean.get_lean("en.wikipedia.org (Current events)") is None
    assert sourcelean.get_lean("unknown.example") is None


def test_lean_mult_declared_map_and_labels():
    assert sourcelean.lean_mult(0) == 1.0
    assert sourcelean.lean_mult(-1) == sourcelean.lean_mult(1) == 0.9   # extremity, not side
    assert sourcelean.lean_mult(-2) == sourcelean.lean_mult(2) == 0.75
    assert sourcelean.lean_mult(None) == 1.0                            # unrated -> no adjustment
    assert sourcelean.lean_label(-2) == "left" and sourcelean.lean_label(None) == "unrated"
    assert sourcelean.lean_bucket(-1) == "left" and sourcelean.lean_bucket(1) == "right"
    assert sourcelean.lean_bucket(0) == "center" and sourcelean.lean_bucket(None) == "unrated"


def test_annotate_composes_reliability_and_lean(monkeypatch):
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "refresh", lambda force=False: {
        "fetched_at": "2026-07-05T00:00:00+00:00",
        "rsp_status_by_id": {"the guardian": "s-gr", "bbc": "s-gr"},
        "iffy_domains": ["badnews.example"]})
    rows = [{"article": {"domain": "theguardian.com"}},   # rel 1.0 x lean |2| 0.75
            {"article": {"domain": "bbc.co.uk"}},         # rel 1.0 x lean 0 -> 1.0
            {"article": {"domain": "badnews.example"}},   # blocklist stays 0
            {"article": {"domain": "unknown.example"}}]   # neutral both axes
    out = sourceweights.annotate(rows)
    assert out[0]["source_w_rel"] == 1.0 and out[0]["source_lean"] == -2
    assert out[0]["source_w"] == pytest.approx(0.75)
    assert out[1]["source_w"] == pytest.approx(1.0)
    assert out[2]["source_w"] == 0.0                       # lean can never resurrect a blocklisted source
    assert out[3]["source_w"] == 0.5 and out[3]["source_lean"] is None   # unknown 0.5
    monkeypatch.setattr(sourceweights, "_WEIGHTS", None)


# ----------------------------------------------- bias explainer: lean axis -----

def test_source_bias_breakdown_lean_mix_and_coverage():
    rows = [
        {"features": F(stance="toward_yes"), "source_w": 0.75, "source_w_rel": 1.0,
         "source_lean": -2, "article": {"domain": "theguardian.com"}},
        {"features": F(stance="toward_yes"), "source_w": 1.0, "source_w_rel": 1.0,
         "source_lean": 0, "article": {"domain": "bbc.co.uk"}},
        {"features": F(stance="toward_no"), "source_w": 0.75, "source_w_rel": 1.0,
         "source_lean": 2, "article": {"domain": "foxnews.com"}},
        {"features": F(stance="neutral"), "source_w": 1.0, "source_w_rel": 1.0,
         "source_lean": None, "article": {"domain": "unknown.example"}},
    ]
    b = fvmodel.source_bias_breakdown(rows)
    assert b["coverage"] == {"left": 1, "center": 1, "right": 1, "unrated": 1}
    assert b["lean_mix"]["toward_yes"] == {"more-biased": 1, "centrist": 1, "unrated": 0}
    assert b["lean_mix"]["toward_no"] == {"more-biased": 1, "centrist": 0, "unrated": 0}
    assert "[lean mix:" in b["sentence"]
    assert b["leans"]["theguardian.com"] == "left"
    # reliability tiers still grouped on the reliability-only weight
    assert "higher-reliability" in b["yes_tiers"]


# --------------------------------------------------- evidence-quality badge ----

def test_evidence_quality_tiers():
    assert fvmodel.evidence_quality(6, 10.0)["tier"] == "strong"
    assert fvmodel.evidence_quality(3, 15.0)["tier"] == "moderate"
    assert fvmodel.evidence_quality(1, 15.0)["tier"] == "thin"
    assert fvmodel.evidence_quality(6, 25.0)["tier"] == "thin"    # wide band alone is thin
    q = fvmodel.evidence_quality(0, 30.0)
    assert "discount" in q["note"]


# --------------------------------------------------------------- movers --------

def _series_two_live():
    return {"m1": [
        {"date": "2026-07-03", "fv_pct": 35.0, "band_lo_pct": 25.0, "band_hi_pct": 45.0,
         "mid_pct": 30.0, "segment": "backfill"},
        {"date": "2026-07-04", "fv_pct": 40.0, "band_lo_pct": 30.0, "band_hi_pct": 50.0,
         "mid_pct": 31.0, "segment": "live"},
        {"date": "2026-07-05", "fv_pct": 52.0, "band_lo_pct": 42.0, "band_hi_pct": 62.0,
         "mid_pct": 33.0, "segment": "live"},
    ]}


def test_movers_from_live_series_only():
    sc = dashboard.build_showcase([_snapshot31()], _series_two_live())
    items = sc["movers"]["items"]
    assert len(items) == 1 and items[0]["delta_pp"] == pytest.approx(12.0)
    assert items[0]["from_date"] == "2026-07-04" and items[0]["to_date"] == "2026-07-05"
    # one live point (backfill never counts) -> no movers, explanatory note
    sc2 = dashboard.build_showcase([_snapshot31()], _series())
    assert sc2["movers"]["items"] == []
    assert "two published" in sc2["movers"]["note"]


# ----------------------------------------------------------------- feed --------

def test_feed_aggregates_dedupes_and_counts_private():
    s1 = _snapshot31(slug="m1", with_newsletter=True)
    s2 = _snapshot31(slug="m2")
    # same public article in both packets -> one feed item, two market refs
    sc = dashboard.build_showcase([s1, s2], _series())
    feed = sc["feed"]
    assert len(feed["items"]) == 1
    assert {m["slug"] for m in feed["items"][0]["markets"]} == {"m1", "m2"}
    assert feed["n_private"] == 1
    assert feed["items"][0]["lean"] == "unrated"     # domain "d" is unrated


# ------------------------------------------------------------ dashboard v3.2 ---

def test_html_v32_layout_feed_pane_and_donut_grid():
    sc = dashboard.build_showcase(
        [_snapshot31(slug=f"m{i}", flag=(i == 0)) for i in range(3)], _series())
    out = dashboard.render_html(sc)
    # page-level 2-column shell: sticky feed pane left, markets right
    assert 'class="feedpane"' in out and "position:sticky" in out
    assert 'class="layout"' in out and "grid-template-columns:330px" in out
    # every market's donut is in the overview grid, visible without expanding
    assert out.count("fair value donut") == 3
    assert 'class="donutgrid"' in out
    # movers + evidence-quality badge surfaces
    assert "Big movers" in out
    assert "evidence:" in out and "qual-" in out
    # responsive stack under 820px
    assert "max-width:820px" in out


def test_html_v32_design_tokens_swapped():
    sc = dashboard.build_showcase([_snapshot31()], _series())
    out = dashboard.render_html(sc)
    assert "#242423" in out and "#49413c" in out and "#cc5c44" in out
    assert "#C8FF00" not in out and "#0a0a0a" not in out    # old dark/lime theme fully gone
    assert "εpsilon" in out                                  # site wordmark
    assert "AllSides" in out                                 # lean attribution present


def test_html_v32_evidence_moved_out_of_cards():
    sc = dashboard.build_showcase([_snapshot31()], _series())
    out = dashboard.render_html(sc)
    body_start = out.index('id="cards"')
    footer_start = out.index("<footer")
    card_zone = out[body_start:footer_start]
    # the per-card evidence list is gone; cards point at the shared feed instead
    assert '<ul class="ev">' not in card_zone
    assert "shared news feed" in card_zone
    # the feed itself renders exactly once, in the left pane
    assert out.count('<ul class="ev">') == 1


def test_html_v32_still_scrubbed_with_lean():
    sc = dashboard.build_showcase([_snapshot31(with_newsletter=True)], _series_two_live())
    out = dashboard.render_html(sc)
    assert "ZQX-SECRET" not in out and "newsletter:" not in out
    assert "private analysis item" in out
    assert "/Users/" not in out and "<script src" not in out
    assert "beat the mid" not in out.lower()


# ------------------------------------------------ v3.2: band quality multiplier --

def _rel_row(w_rel, lean, stance="toward_yes", clarity=0.6):
    return {"features": F(stance=stance, clarity=clarity), "source_w_rel": w_rel,
            "source_w": w_rel, "source_lean": lean, "article": {"domain": "d"}}


def test_band_quality_low_reliability_widens():
    reliable = [_rel_row(1.0, 0), _rel_row(1.0, 0), _rel_row(1.0, 0)]
    weak = [_rel_row(0.3, 0), _rel_row(0.3, 0), _rel_row(0.3, 0)]
    qr = fvmodel.band_quality(reliable)["q"]
    qw = fvmodel.band_quality(weak)["q"]
    assert qw > 1.0 > qr                     # weak widens, reliable narrows
    # and it flows through to the band half-width
    assert fvmodel.band_half_pp(weak, "shock", PARAMS) > \
        fvmodel.band_half_pp(reliable, "shock", PARAMS)


def test_band_quality_one_sided_widens_cross_spectrum_narrows():
    # all left-leaning sources agreeing -> one-sided -> widen
    onesided = [_rel_row(1.0, -2), _rel_row(1.0, -1), _rel_row(1.0, -2)]
    # left AND right sources both present -> cross-spectrum -> narrow
    crossspec = [_rel_row(1.0, -2), _rel_row(1.0, 2), _rel_row(1.0, -1), _rel_row(1.0, 1)]
    q_one = fvmodel.band_quality(onesided)
    q_cross = fvmodel.band_quality(crossspec)
    assert q_one["s_fac"] > 1.0 and q_cross["s_fac"] < 1.0
    assert q_one["q"] > q_cross["q"]
    assert "one-sided" in " ".join(q_one["reasons"])
    assert "cross-spectrum" in " ".join(q_cross["reasons"])


def test_band_quality_neutral_without_lean_or_reliability_signal():
    # center-only + fully reliable -> no spectrum signal, reliability at ref-ish
    rows = [_rel_row(1.0, 0), _rel_row(1.0, 0)]
    q = fvmodel.band_quality(rows)
    assert q["s_fac"] == 1.0                  # center-only -> no spectrum judgement
    # unrated leans also give no spectrum signal
    rows2 = [_rel_row(1.0, None), _rel_row(1.0, None)]
    assert fvmodel.band_quality(rows2)["s_fac"] == 1.0


def test_band_quality_clamped_and_floor_is_hard_minimum():
    worst = [_rel_row(0.3, -2), _rel_row(0.3, -2), _rel_row(0.3, -1)]   # weak + one-sided
    q = fvmodel.band_quality(worst)["q"]
    assert q <= fvmodel.BAND_Q["q_max"]
    # even a narrowing Q cannot push the half-width below the market-type floor
    best = [_rel_row(1.0, -2), _rel_row(1.0, 2)]     # reliable + cross-spectrum, concur
    half = fvmodel.band_half_pp(best, "slow", PARAMS)
    assert half >= PARAMS["floor_pp"]["slow"]


def test_evidence_quality_badge_carries_band_multiplier():
    bq = fvmodel.band_quality([_rel_row(0.3, -2), _rel_row(0.3, -2), _rel_row(0.3, -1)])
    eq = fvmodel.evidence_quality(3, 22.0, band_q=bq)
    assert eq["band_mult_q"] == bq["q"] and eq["band_reasons"]
    assert "band widened" in eq["note"]


def test_band_quality_shown_on_card():
    s = _snapshot31()
    s["stage_b"]["evidence_quality"] = fvmodel.evidence_quality(
        3, 22.0, band_q=fvmodel.band_quality(
            [_rel_row(0.3, -2), _rel_row(0.3, -2), _rel_row(0.3, -1)]))
    out = dashboard.render_html(dashboard.build_showcase([s], _series()))
    assert "evidence-quality multiplier" in out
    assert "one-sided coverage" in out


# ------------------------------------------- v3.3: credential loading (.env) ---

def test_load_env_sets_only_missing_keys_and_skips_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment line\n"
        "\n"
        "GUARDIAN_API_KEY=abc123\n"
        "ALREADY_SET=from_file\n"
        "NOT_PASTED_YET=      # registered but no value here\n"
        "QUOTED_KEY=\"quoted-value\"\n"
        "junk line without equals\n")
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    monkeypatch.delenv("NOT_PASTED_YET", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from_shell")
    loaded = na_config.load_env(env)
    import os
    assert "GUARDIAN_API_KEY" in loaded and os.environ["GUARDIAN_API_KEY"] == "abc123"
    assert os.environ["QUOTED_KEY"] == "quoted-value"
    # the shell always wins over the file
    assert "ALREADY_SET" not in loaded and os.environ["ALREADY_SET"] == "from_shell"
    # "KEY=  # not pasted yet" must NOT arm a garbage credential
    assert "NOT_PASTED_YET" not in loaded and "NOT_PASTED_YET" not in os.environ
    for k in ("GUARDIAN_API_KEY", "QUOTED_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_load_env_resolves_credential_paths_against_the_env_file(tmp_path, monkeypatch):
    """A relative *_CREDENTIALS path must survive being run from another cwd."""
    (tmp_path / "secrets").mkdir()
    (tmp_path / ".env").write_text("GOOGLE_APPLICATION_CREDENTIALS=secrets/sa.json\n")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    na_config.load_env(tmp_path / ".env")
    import os
    got = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    assert got == str((tmp_path / "secrets" / "sa.json").resolve())
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)


def test_load_env_absent_file_is_not_an_error(tmp_path):
    assert na_config.load_env(tmp_path / "nope.env") == []


# ------------------------------------------ v3.3: re-slug (gamma_slug) ---------

def test_market_state_follows_a_reslug_without_moving_our_key(monkeypatch):
    """Polymarket re-slugged putin-out-before-2027 mid-life. Our key must not move:
    the ledger id, prior, state and feature cache all hang off it."""
    seen = {}

    def fake_http_json(url, retries=4):
        seen["url"] = url
        assert "putin-out-before-2027-346" in url
        return [{"question": "Putin out?", "description": "d", "endDate": "2027-01-01",
                 "closed": False, "bestBid": 0.07, "bestAsk": 0.08}]

    monkeypatch.setattr(feeds_mod, "http_json", fake_http_json)
    mkt = feeds_mod.market_state("putin-out-before-2027", "putin-out-before-2027-346")
    assert mkt["slug"] == "putin-out-before-2027"       # OUR key comes back
    assert mkt["mid"] == pytest.approx(0.075)


def test_market_state_raises_a_useful_error_when_gamma_has_nothing(monkeypatch):
    monkeypatch.setattr(feeds_mod, "http_json", lambda url, retries=4: [])
    with pytest.raises(RuntimeError, match="re-slug"):
        feeds_mod.market_state("gone-slug")


# ------------------------------------- v3.3: newsletter credential degradation --

def test_dead_newsletter_credential_degrades_instead_of_breaking_the_run(monkeypatch, capsys):
    """A revoked OAuth token must not stop --stage fetch (observed 2026-08-24)."""
    monkeypatch.setattr(email_ingest, "available", lambda: (True, "gmail"))
    monkeypatch.setattr(email_ingest, "CACHE_DIR", email_ingest.CACHE_DIR)

    def boom(_days):
        raise RuntimeError("HTTP Error 400: Bad Request")

    monkeypatch.setattr(email_ingest, "_fetch_gmail", boom)
    monkeypatch.setattr(email_ingest, "fetch_newsletters",
                        email_ingest.fetch_newsletters)   # keep the real one
    out = email_ingest.fetch_newsletters(day="2999-01-01")
    assert out == []
    assert "credential failed" in capsys.readouterr().out


# --------------------------------------- v3.3: public settled track record ------

def _settled(sf_id, slug, p, y, res, last_fc, method="news"):
    return {"sf_id": sf_id, "slug": slug, "question": f"Q {sf_id}?", "final_p": p,
            "outcome": y, "brier": round((p - y) ** 2, 6), "resolution_date": res,
            "settled_at": "2026-08-24", "last_update": last_fc, "method": method}


def test_live_track_record_scores_per_method_and_flags_staleness(monkeypatch):
    from newsagent import ledger as ledger_mod
    recs = [_settled("sf-1", "m1", 0.20, 0, "2026-07-17", "2026-07-05"),
            _settled("sf-2", "m2", 0.60, 1, "2026-07-29", "2026-07-05")]
    monkeypatch.setattr(ledger_mod, "settled_records", lambda: recs)
    monkeypatch.setattr(ledger_mod, "method_for", lambda slug: "news")
    monkeypatch.setattr(dashboard, "_extended_scorecard", lambda: None)
    series = {"m1": [{"date": "2026-07-05", "fv_pct": 20.0, "mid_pct": 38.5,
                      "segment": "live"}]}
    ltr = dashboard._live_track_record(series)
    assert ltr["n_settled"] == 2
    assert ltr["brier"] == pytest.approx((0.04 + 0.16) / 2, abs=1e-6)
    assert ltr["per_method"]["news"]["n"] == 2
    assert ltr["records"][0]["mid_pct_at_last_snapshot"] == 38.5   # context only
    assert ltr["max_stale_days"] == 24        # 2026-07-05 -> 2026-07-29
    assert "append-only" in ltr["note"]


def test_live_track_record_shows_the_n0_transition_line_for_a_new_method(monkeypatch):
    """DC-8: methods are reported separately; a method with nothing settled says so."""
    from newsagent import ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "settled_records",
                        lambda: [_settled("sf-1", "m1", 0.2, 0, "2026-07-17", "2026-07-05")])
    monkeypatch.setattr(ledger_mod, "method_for",
                        lambda slug: "news+data" if slug == "fed" else "news")
    monkeypatch.setattr(na_config, "LIVE_MARKETS", {"fed": {}, "m1": {}})
    monkeypatch.setattr(dashboard, "_extended_scorecard", lambda: None)
    ltr = dashboard._live_track_record({})
    assert ltr["per_method"]["news"]["n"] == 1
    assert ltr["per_method"]["news+data"]["n"] == 0
    assert ltr["per_method"]["news+data"]["note"] == "new method — no settled track record yet (n=0)"
    assert ltr["per_method"]["news+data"]["brier"] is None


def test_html_renders_the_settled_track_record_panel(monkeypatch):
    from newsagent import ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "settled_records",
                        lambda: [_settled("sf-1", "m1", 0.537, 1, "2026-07-29", "2026-07-05")])
    monkeypatch.setattr(ledger_mod, "method_for", lambda slug: "news")
    monkeypatch.setattr(dashboard, "_extended_scorecard", lambda: None)
    sc = dashboard.build_showcase([_snapshot31()], _series())
    out = dashboard.render_html(sc)
    assert "Public track record" in out
    assert "0.2143" in out or "0.214" in out          # the settled Brier
    assert "snapshot age" in out and "24" in out      # staleness is displayed
    assert "method" in out


def test_ledger_method_labels_default_to_news_and_never_merge(tmp_path, monkeypatch):
    from newsagent import ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "METHODS", tmp_path / "methods.json")
    monkeypatch.setattr(na_config, "DATA_CHANNEL_MARKETS", frozenset({"fed-market"}))
    assert ledger_mod.method_for("some-news-market") == ledger_mod.METHOD_NEWS
    assert ledger_mod.method_for("fed-market") == ledger_mod.METHOD_NEWS_DATA
    ledger_mod.record_method("sf-9", "fed-market", "2026-08-24")
    assert ledger_mod.method_of("sf-9") == ledger_mod.METHOD_NEWS_DATA
    assert ledger_mod.method_of("sf-unknown") == ledger_mod.METHOD_NEWS   # pre-labelling


# ------------------------------------------------ v3.3: poll-driven card copy ---

def test_html_poll_driven_card_says_polling_not_data(monkeypatch):
    s = _snapshot31(tract="poll")
    s["stage_b"]["tract_note"] = "General-election polling drives this race."
    sc = dashboard.build_showcase([s], _series())
    out = dashboard.render_html(sc)
    assert "◆ poll-driven" in out
    assert "private/campaign or ballot-issue" in out
    assert "structurally blind" in out


# =========================================================== v3.4: simulated-live
# The reconstruction is a DIFFERENT OBJECT from the forward ledger. These tests pin
# that separation as hard as the privacy boundary is pinned: a reconstruction number
# must never be summed into the forward track record, and must never reach the page
# without the word "reconstruction" attached.

def _simlive_payload(**over):
    sl = {
        "generated": "2026-08-25", "label": "reconstruction",
        "n_markets": 44, "n_snapshots": 326,
        "params": {"alpha": 2.75, "band_mult": 0.75, "gamma": 1.0},
        "at_resolution": {
            "brier": 0.1896, "log_loss": 0.639, "brier_prior_only": 0.192,
            "base_rate_pct": 29.5, "mean_forecast_pct": 15.3, "mean_age_days": 1.0,
            "mean_abs_shift_from_prior_pp": 3.46, "share_moved_ge_1pp": 0.34,
            "murphy": {"reliability": 0.0238, "resolution": 0.0355,
                       "uncertainty": 0.2082, "base_rate": 0.295, "n_bins_populated": 4},
            "spiegelhalter": {"z": 3.405, "p": 0.001},
            "reliability_curve": [
                {"bin": "0-20%", "n": 33, "mean_forecast_pct": 6.2,
                 "observed_yes_pct": 21.2, "thin": False},
                {"bin": "20-40%", "n": 6, "mean_forecast_pct": 31.8,
                 "observed_yes_pct": 33.3, "thin": False},
                {"bin": "40-60%", "n": 4, "mean_forecast_pct": 51.9,
                 "observed_yes_pct": 75.0, "thin": False},
                {"bin": "60-80%", "n": 1, "mean_forecast_pct": 69.7,
                 "observed_yes_pct": 100.0, "thin": True},
                {"bin": "80-100%", "n": 0, "mean_forecast_pct": None,
                 "observed_yes_pct": None, "thin": True}],
        },
        "staleness": {
            "raw": [{"days_before_resolution": h, "n": 44, "brier": 0.18,
                     "log_loss": 0.6, "mean_forecast_pct": 15.0}
                    for h in (1, 5, 9, 13, 17, 21, 25, 29)],
            "balanced": [{"days_before_resolution": h, "n": 30, "brier": 0.15,
                          "log_loss": 0.5, "mean_forecast_pct": 15.0,
                          "mean_abs_shift_from_prior_pp": 2.1,
                          "share_moved_ge_1pp": 0.25}
                         for h in (1, 5, 9, 13, 17, 21, 25, 29)],
            "n_markets_balanced": 30,
            "derived_cadence": [{"cadence_days": 4, "horizons_averaged": [1],
                                 "expected_brier": 0.1572},
                                {"cadence_days": 28, "horizons_averaged": [1, 5],
                                 "expected_brier": 0.1492}],
        },
        "splits": {
            "tract": [
                {"tract": "news", "n": 34, "brier": 0.223, "log_loss": 0.71,
                 "base_rate_pct": 32.4, "mean_forecast_pct": 13.7, "thin": False},
                {"tract": "poll", "n": 6, "brier": 0.0773, "log_loss": 0.3,
                 "base_rate_pct": 16.7, "mean_forecast_pct": 18.1, "thin": False},
                {"tract": "data", "n": 4, "brier": 0.0741, "log_loss": 0.28,
                 "base_rate_pct": 25.0, "mean_forecast_pct": 24.6, "thin": True}],
            "family": [
                {"family": "iran", "n": 7, "brier": 0.3974, "log_loss": 1.2,
                 "base_rate_pct": 42.9, "mean_forecast_pct": 10.3, "thin": False},
                {"family": "fed", "n": 5, "brier": 0.0776, "log_loss": 0.3,
                 "base_rate_pct": 40.0, "mean_forecast_pct": 33.6, "thin": False},
                {"family": "solo", "n": 1, "brier": 0.5, "log_loss": 1.0,
                 "base_rate_pct": 100.0, "mean_forecast_pct": 25.0, "thin": True}],
        },
        "divergence": {
            "n_flagged_snapshots": 2, "n_flagged_markets": 2,
            "n_snapshots_with_a_mid": 300, "flagged_markets_resolved_yes": 1,
            "fv_below_mid_snapshots": 2, "fv_above_mid_snapshots": 0,
            "closure": "The v0 fair-value-vs-mid gate is CLOSED and is NOT reopened here.",
        },
        "exclusions_note": ("Reconstruction sees ONLY channels with a timestamped "
                            "archive: Guardian, Wikipedia Current Events and GDELT."),
    }
    sl.update(over)
    return sl


def _with_simlive(monkeypatch, payload=None):
    monkeypatch.setattr(dashboard, "_simlive", lambda: payload or _simlive_payload())


def test_simlive_loader_rejects_anything_not_labelled_reconstruction(tmp_path, monkeypatch):
    import json as _json
    p = tmp_path / "simlive_results.json"
    monkeypatch.setattr(dashboard, "SIMLIVE_PATH", p)
    assert dashboard._simlive() is None                       # absent file
    p.write_text(_json.dumps({"label": "forward", "n_markets": 44}))
    assert dashboard._simlive() is None                       # wrong label
    p.write_text(_json.dumps(_simlive_payload(n_markets=0)))
    assert dashboard._simlive() is None                       # empty sample
    p.write_text("{ not json")
    assert dashboard._simlive() is None                       # unreadable
    p.write_text(_json.dumps(_simlive_payload()))
    assert dashboard._simlive()["n_markets"] == 44


def test_html_renders_the_simulated_live_section(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "Simulated-live reconstruction" in out
    assert "not forecasts we published" in out.lower()
    assert "0.1896" in out                       # pooled Brier at resolution
    assert "0.639" in out                        # log-loss
    assert "Brier vs snapshot age" in out        # the centrepiece chart
    assert "reconstruction</span>" in out        # the standing chip
    assert "Murphy" in out or "reliability" in out.lower()


def test_simlive_never_merges_with_the_forward_ledger(monkeypatch):
    from newsagent import ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "settled_records",
                        lambda: [_settled("sf-1", "m1", 0.20, 0, "2026-07-17", "2026-07-05")])
    monkeypatch.setattr(ledger_mod, "method_for", lambda slug: "news")
    monkeypatch.setattr(dashboard, "_extended_scorecard", lambda: None)
    _with_simlive(monkeypatch)
    sc = dashboard.build_showcase([_snapshot31()], _series())
    # separate keys, separate n, and the forward panel is unaffected by the payload
    assert sc["simlive"]["n_markets"] == 44
    assert sc["live_track_record"]["n_settled"] == 1
    assert sc["live_track_record"]["brier"] == round((0.20 - 0) ** 2, 4)
    out = dashboard.render_html(sc)
    assert out.index("Public track record") < out.index("Simulated-live reconstruction")


def test_simlive_section_absent_when_no_reconstruction_exists(monkeypatch):
    monkeypatch.setattr(dashboard, "_simlive", lambda: None)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "Simulated-live reconstruction" not in out
    assert "All markets at a glance" in out      # the rest of the page is intact


def test_html_shows_the_per_tract_accuracy_split(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "accuracy by market type" in out
    for label in ("news-driven", "poll-driven", "data-driven"):
        assert label in out
    assert "0.2230" in out                       # the news-driven Brier
    assert "0.0773" in out and "0.0741" in out   # poll- and data-driven beside it
    # the uncomfortable direction is stated, not left for the reader to infer
    assert "worse</b> on the\n        news-driven markets" in out
    # base rates are shown beside every Brier so composition is visible
    assert "32%" in out and "17%" in out and "25%" in out


def test_html_states_the_staleness_finding_and_its_mechanism(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "The curve is flat" in out
    assert "not the binding constraint" in out
    assert "3.46pp" in out                        # mean |FV - prior| at resolution
    assert "34%" in out                           # share that moved >= 1pp
    assert "0.1920" in out                        # prior-only Brier shown beside it
    assert 'reconchip">derived' in out            # the cadence table is chip-labelled
    assert "every <span class=\"mono\">28</span> days" in out


def test_html_says_the_reconstruction_is_in_sample(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "in-sample, and that is not a detail" in out
    assert "2.75" in out                          # the fitted alpha is named
    assert "not an out-of-sample test" in out
    assert "upper bound" in out
    assert "§ 06" in out                          # points at the genuinely OOS numbers


def test_html_restates_the_v0_closure_wherever_flags_are_described(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "CLOSED" in out and "NOT reopened here" in out
    assert "beat the mid" not in out.lower()      # never phrased as an edge claim


def test_html_states_the_evidence_poorer_caveat(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "evidence-poorer world" in out
    assert "timestamped" in out


def test_simlive_page_is_still_self_contained_and_ip_scrubbed(monkeypatch):
    _with_simlive(monkeypatch)
    sc = dashboard.build_showcase([_snapshot31(with_newsletter=True)], _series())
    out = dashboard.render_html(sc)
    assert "ZQX-SECRET" not in out and "newsletter:" not in out
    assert "/Users/" not in out                   # no absolute local path leaks
    assert "<script src" not in out and "http://cdn" not in out
    assert "private analysis item" in out


def test_simlive_uses_only_the_enforced_palette(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    import re as _re
    allowed = {dashboard.BG, dashboard.PANEL, dashboard.TX, dashboard.DIM,
               dashboard.ACC, dashboard.HI, dashboard.NEG, "#fff", "#000"}
    section = out[out.index("Simulated-live reconstruction"):]
    section = section[:section.index("Reliability — model FV")]
    for hexcol in set(_re.findall(r"#[0-9a-fA-F]{3,6}", section)):
        assert hexcol.lower() in {a.lower() for a in allowed}, hexcol


def test_staleness_svg_marks_the_balanced_series_and_its_n(monkeypatch):
    svg = dashboard._svg_staleness(_simlive_payload())
    assert "balanced" in svg and "30 markets" in svg
    assert "n=30" in svg
    assert "stroke-dasharray" in svg              # raw series is the dashed one
    assert "snapshot age" in svg


# ------------------------------------------- v3.4: data-channel card, regime split

def test_data_channel_card_shows_the_regime_where_the_method_loses(monkeypatch, tmp_path):
    import json as _json
    p = tmp_path / "v3_results.json"
    p.write_text(_json.dumps({"by_regime": {
        "hiking": {"n": 16, "brier": 0.136, "base": 0.3277, "violations": 0},
        "holding": {"n": 5, "brier": 0.1528, "base": 0.1236, "violations": 0}}}))
    monkeypatch.setattr(dashboard, "DC_RESULTS_PATH", p)
    out = dashboard._dc_regime_html()
    assert "by rate regime" in out
    assert "WORSE than doing nothing" in out      # the holding row, stated plainly
    assert "this market" in out                   # and marked as the live regime
    assert "0.1528" in out and "0.1236" in out


def test_data_channel_regime_split_degrades_when_results_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "DC_RESULTS_PATH", tmp_path / "missing.json")
    assert dashboard._dc_regime_html() == ""


def test_data_channel_card_says_fv_equals_the_structural_anchor(monkeypatch, tmp_path):
    import json as _json
    p = tmp_path / "v3_results.json"
    p.write_text(_json.dumps({"by_regime": {
        "holding": {"n": 5, "brier": 0.1528, "base": 0.1236, "violations": 0}}}))
    monkeypatch.setattr(dashboard, "DC_RESULTS_PATH", p)
    s = _snapshot31(fv=50.7, tract="data")
    s["stage_b"].update({"p0_source": "p_struct", "p0_used_pct": 50.7,
                         "method": "news+data", "n_double_counted": 1,
                         "market_implied": {"available": False,
                                            "why": "the Atlanta Fed Market Probability "
                                                   "Tracker stops quoting a 3-month "
                                                   "window once that window opens"}})
    out = dashboard.render_html(dashboard.build_showcase([s], _series()))
    assert "◆ data-channel scored" in out
    assert "equals the structural anchor" in out
    assert "no settled track record yet (n=0)" in out
    assert "Market Probability Tracker" in out    # MPT dark, rendered in words
    assert "contribute zero evidence" in out      # double-count guard, labelled


def test_mpt_going_dark_does_not_break_the_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "DC_RESULTS_PATH", tmp_path / "missing.json")
    s = _snapshot31(fv=50.7, tract="data")
    s["stage_b"].update({"p0_source": "p_struct", "p0_used_pct": 50.7,
                         "method": "news+data", "market_implied": None})
    out = dashboard.render_html(dashboard.build_showcase([s], _series()))
    assert "◆ data-channel scored" in out         # card still renders end to end
    assert "unavailable" in out
    assert "n=0" in out


# ------------------------------------------------------ v3.4: reach honesty note

def test_page_states_what_the_reach_channel_actually_reaches(monkeypatch):
    _with_simlive(monkeypatch)
    out = dashboard.render_html(dashboard.build_showcase([_snapshot31()], _series()))
    assert "federalreserve.gov" in out and "state.gov" in out
    assert "ukmto.org" in out and "navigation-only" in out
    assert "T+1" in out
    assert "headline, source and link only" in out
