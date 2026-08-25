"""Tests for the agent-reach evidence extension (v3.4).

Pure-function tests — no network, no agent-reach binary required. Every fixture
below is a REAL payload shape measured live on 2026-08-24 during the build:
the r.jina.ai response envelope, the reuters.com abuse-block JSON, the state.gov
404-served-as-200, the yt-dlp metadata for the July FOMC press conference.

Run: ``PYTHONPATH=. uv run pytest tests/test_newsagent_reach.py``
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from newsagent import dashboard, feeds, fvmodel, reach, sourceweights

CUTOFF = datetime(2026, 8, 24, 23, 59, 59, tzinfo=timezone.utc)


# ------------------------------------------------------------------ fixtures ---

def jina(title="Federal Reserve issues FOMC statement",
         published="Wed, 29 Jul 2026 18:00:15 GMT",
         body=None, url="https://www.federalreserve.gov/x.htm") -> str:
    """A well-formed r.jina.ai response, in the exact header shape it returns."""
    body = body or (
        "The Federal Open Market Committee approved the following statement for "
        "release by a 9 - 3 vote:\n"
        "The Committee decided to maintain the target range for the federal funds "
        "rate at 3-1/2 to 3-3/4 percent, in support of the Federal Reserve's dual "
        "mandate. The Committee is continuing its policy of maintaining ample "
        "reserves in the banking system.\n"
        "Economic activity is expanding at a solid pace despite elevated "
        "uncertainty that owes, in part, to the conflict in the Middle East.\n"
        "Inflation remains elevated relative to the Committee's 2 percent goal, in "
        "part reflecting supply shocks that have driven price increases in certain "
        "sectors, including energy. The Committee will deliver price stability.\n")
    head = f"Title: {title}\n\nURL Source: {url}\n\n"
    if published:
        head += f"Published Time: {published}\n\n"
    return head + "Markdown Content:\n" + body


def doc_job(**over) -> dict:
    job = {"id": "j1", "role": "document", "label": "FOMC statement",
           "domain": "federalreserve.gov", "source_type": "official_primary",
           "immutable": True, "slugs": ["fed-sep"],
           "timestamp_class": "verified", "timestamp_trusted": True,
           "url": "https://www.federalreserve.gov/x.htm"}
    job.update(over)
    return job


FOMC_INFO = {   # measured live 2026-08-24 from yt-dlp on video DLFXUkOc_7I
    "id": "DLFXUkOc_7I", "title": "FOMC Press Conference, July 29, 2026",
    "channel": "Federal Reserve", "uploader_id": "@federalreserve",
    "channel_id": "UCAzhpt9DmG6PnHXjmJTvRGQ", "timestamp": 1785356671,
    "release_timestamp": None, "duration": 2704, "live_status": "not_live",
    "webpage_url": "https://www.youtube.com/watch?v=DLFXUkOc_7I",
}


def vtt(n_words=400, text=None) -> str:
    line = text or " ".join(f"word{i}" for i in range(n_words))
    return ("WEBVTT\nKind: captions\nLanguage: en\n\n"
            "1\n00:00:00.000 --> 00:00:04.000\n" + line + "\n\n"
            "2\n00:00:04.000 --> 00:00:08.000\n<c>tail</c> of the transcript\n")


def video_job(**over) -> dict:
    job = {"id": "v1", "role": "video", "label": "FOMC presser",
           "domain": "youtube.com/@federalreserve",
           "source_type": "official_transcript", "slugs": ["fed-sep"]}
    job.update(over)
    return job


# ------------------------------------------------------------- chrome strip ----

def test_chrome_strip_keeps_prose_and_drops_navigation():
    nav = "\n".join(f"*   [Nav item {i}](https://x.gov/{i})" for i in range(40))
    raw = jina(body=nav + "\n\n" + "The Committee decided to maintain the target "
                                   "range for the federal funds rate at 3-1/2 to "
                                   "3-3/4 percent in support of its dual mandate.\n")
    out = reach.strip_chrome(raw)
    assert "decided to maintain the target range" in out
    assert "Nav item 12" not in out


def test_chrome_strip_drops_gov_boilerplate_and_link_bars():
    raw = jina(body=(
        "An official website of the United States Government\n"
        "**Official websites use .gov**\n"
        "A **lock ()** or **https://** means you've safely connected to the .gov "
        "website. Share sensitive information only on official, secure websites.\n"
        "[A](https://a.gov)[B](https://b.gov)[C](https://c.gov)[D](https://d.gov)\n"
        "The Committee decided to maintain the target range for the federal funds "
        "rate at 3-1/2 to 3-3/4 percent, in support of its dual mandate.\n"))
    out = reach.strip_chrome(raw)
    assert "decided to maintain" in out
    assert "official website of the United States Government" not in out
    assert "Share sensitive information" not in out


def test_chrome_strip_is_the_reason_the_gate_is_affordable():
    """4.3% signal on the measured FOMC page — the strip is mandatory, not cosmetic."""
    nav = "\n".join(f"*   [Menu entry {i}](https://x.gov/{i})" for i in range(400))
    raw = jina(body=nav + "\n\nThe Committee decided to maintain the target range "
                          "for the federal funds rate at 3-1/2 to 3-3/4 percent.\n")
    assert len(reach.strip_chrome(raw)) < 0.2 * len(raw)


# ------------------------------------------------------- per-item validation ---

def test_jina_json_error_envelope_is_rejected():
    """The reuters.com case: HTTP 403 with a well-formed JSON body."""
    raw = json.dumps({"code": 403, "status": 42206,
                      "message": "AbuseAlleviationError: Anonymous access to domain "
                                 "www.reuters.com blocked until Mon Aug 24 2026"})
    parsed, why = reach.parse_jina(raw)
    assert parsed is None and "403" in why


def test_jina_404_served_inside_a_200_is_rejected():
    """The state.gov case: HTTP 200 whose body is the origin's 404 page."""
    raw = ("Title: Page Not Found\n\nURL Source: https://www.state.gov/briefings/\n\n"
           "Warning: Target URL returned error 404: Not Found\n\n"
           "Markdown Content:\nThe page you are looking for could not be found. "
           "Please try the search box or return to the home page for more.\n")
    parsed, why = reach.parse_jina(raw)
    assert parsed is None and "404" in why


def test_jina_missing_title_header_is_rejected():
    parsed, why = reach.parse_jina("URL Source: https://x.gov\n\nsome text\n")
    assert parsed is None and "Title" in why


def test_jina_too_short_after_strip_is_rejected():
    parsed, why = reach.parse_jina(jina(body="Short.\n"))
    assert parsed is None and "chrome-strip" in why


def test_a_valid_page_parses_with_its_release_instant():
    parsed, why = reach.parse_jina(jina())
    assert why == ""
    assert parsed["published"] == datetime(2026, 7, 29, 18, 0, 15, tzinfo=timezone.utc)


# -------------------------------------------------------- timestamp discipline --

def test_undated_page_is_absent_not_degraded():
    """nobelprize.org: HTTP 200, real content, no Published Time -> not evidence."""
    item, why = reach.item_from_jina(doc_job(), jina(published=None), CUTOFF)
    assert item is None and "Published Time" in why


def test_rolling_index_page_can_never_become_a_dated_item():
    """fomccalendars.htm: its Last-Modified is a touch time, not an event time."""
    item, why = reach.item_from_jina(doc_job(role="index"), jina(), CUTOFF)
    assert item is None and "touch time" in why


def test_item_after_the_run_cutoff_is_dropped():
    item, why = reach.item_from_jina(
        doc_job(), jina(published="Wed, 30 Sep 2026 18:00:15 GMT"), CUTOFF)
    assert item is None and "after the run cutoff" in why


def test_transcript_is_dated_at_the_later_of_upload_and_stream_end():
    """max(upload, release_timestamp + duration).

    The scoping probe measured a re-stream whose release_timestamp was three
    minutes BEFORE the statement it contained; dating by stream start would let
    post-cutoff content into a pre-cutoff packet.
    """
    info = {**FOMC_INFO, "timestamp": 1785356671,      # 2026-07-29T20:24:31Z
            "release_timestamp": 1785348000, "duration": 2704}
    item, why = reach.item_from_transcript(video_job(), info, vtt(), CUTOFF)
    assert why == "" and item["seendate"] == "20260729T202431Z"

    # stream end later than the upload stamp -> stream end wins
    late = {**FOMC_INFO, "timestamp": 1785348000,
            "release_timestamp": 1785356671, "duration": 3600}
    item2, _ = reach.item_from_transcript(video_job(), late, vtt(), CUTOFF)
    assert item2["seendate"] > "20260729T202431Z"


def test_transcript_never_dated_by_the_event_date_in_its_title():
    info = {**FOMC_INFO, "timestamp": None, "release_timestamp": None}
    item, why = reach.item_from_transcript(video_job(), info, vtt(), CUTOFF)
    assert item is None and "NEVER dated by the event date in its title" in why


def test_still_accruing_stream_is_rejected():
    for status in ("is_live", "post_live", "is_upcoming"):
        item, why = reach.item_from_transcript(
            video_job(), {**FOMC_INFO, "live_status": status}, vtt(), CUTOFF)
        assert item is None and status in why


def test_short_caption_track_is_a_transient_not_evidence():
    item, why = reach.item_from_transcript(video_job(), FOMC_INFO, vtt(n_words=50),
                                           CUTOFF)
    assert item is None and "known transient" in why


def test_vtt_cue_strip_collapses_rolling_duplicates():
    dup = ("WEBVTT\n\n" + "".join(
        f"{i}\n00:00:0{i}.000 --> 00:00:0{i + 1}.000\nthe same line\n\n"
        for i in range(5)))
    assert reach.strip_vtt(dup) == "the same line"


# ------------------------------------------------- Stage-A cache-key collision --

def test_immutable_document_keeps_its_title_verbatim():
    """Official documents DISPLAY their headline — a hash suffix would be public."""
    dt = datetime(2026, 7, 29, tzinfo=timezone.utc)
    assert reach.content_unique_title("FOMC statement", dt, "body", True) \
        == "FOMC statement"


def test_non_immutable_titles_are_content_unique():
    """`features.cache_key` is sha1(slug|title|vN): same title + different content
    would silently return yesterday's features for today's text."""
    from newsagent import features
    dt = datetime(2026, 7, 29, tzinfo=timezone.utc)
    a = reach.content_unique_title("Daily briefing", dt, "content A", False)
    b = reach.content_unique_title("Daily briefing", dt, "content B", False)
    assert a != b
    assert features.cache_key("m", a) != features.cache_key("m", b)


# ------------------------------------------------------------- doctor gating ---

def test_missing_binary_is_a_clean_skip(monkeypatch):
    monkeypatch.setattr(reach.shutil, "which", lambda _: None)
    doc = reach.doctor()
    assert doc["available"] is False and "not installed" in doc["why"]
    ok, why = reach.channel_status(doc, "web")
    assert ok is False and "not installed" in why


def test_doctor_result_is_cached_to_the_day_dir(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class P:
            stdout = json.dumps({"web": {"status": "ok", "message": "",
                                         "active_backend": "Jina Reader"}})
        return P()

    monkeypatch.setattr(reach.shutil, "which", lambda _: "/bin/agent-reach")
    monkeypatch.setattr(reach.subprocess, "run", fake_run)
    reach.doctor(tmp_path)
    reach.doctor(tmp_path)
    assert len(calls) == 1                       # second call served from cache
    assert (tmp_path / "reach_doctor.json").exists()


def test_unhealthy_channel_skips_its_jobs_with_the_reason():
    doc = {"available": True, "channels": {
        "web": {"status": "warn", "message": "configured but unverified"},
        "youtube": {"status": "ok", "message": ""}}}
    ok, why = reach.channel_status(doc, "web")
    assert ok is False and "warn" in why


# ------------------------------------------------------------------ worklist ---

SOURCES = {
    "calendars": {"fomc": ["2026-09-16"]},
    "markets": {
        "fed": [
            {"label": "FOMC statement", "role": "document", "domain": "federalreserve.gov",
             "source_type": "official_primary", "immutable": True, "enabled": True,
             "url": "https://www.federalreserve.gov/newsevents/pressreleases/"
                    "monetary{YYYYMMDD}a.htm",
             "trigger": {"kind": "calendar", "calendar": "fomc", "offsets_days": [0, 1]}},
            {"label": "Dead source", "role": "index", "domain": "iaea.org",
             "source_type": "official_primary", "enabled": False,
             "url": "https://www.iaea.org/x", "trigger": {"kind": "always"},
             "probe": {"date": "2026-08-24", "published_time": "blocked"}},
        ],
        "hormuz-a": [
            {"label": "UKMTO incidents", "role": "index", "domain": "ukmto.org",
             "source_type": "official_primary", "enabled": True,
             "url": "https://www.ukmto.org/recent-incidents",
             "trigger": {"kind": "always"}},
        ],
        "hormuz-b": [
            {"label": "UKMTO incidents", "role": "index", "domain": "ukmto.org",
             "source_type": "official_primary", "enabled": True,
             "url": "https://www.ukmto.org/recent-incidents",
             "trigger": {"kind": "always"}},
        ],
    },
}
OK_DOCTOR = {"available": True, "channels": {"web": {"status": "ok", "message": ""},
                                             "youtube": {"status": "ok", "message": ""}}}


def test_calendar_trigger_fires_on_the_event_and_resolves_the_event_date_url():
    """A T+1 fetch still builds the MEETING-day URL, not tomorrow's."""
    wl = reach.build_worklist("2026-09-17", ["fed"], OK_DOCTOR, SOURCES)
    urls = [j["url"] for j in wl["jobs"]]
    assert "https://www.federalreserve.gov/newsevents/pressreleases/" \
           "monetary20260916a.htm" in urls


def test_trigger_does_not_fire_off_calendar():
    wl = reach.build_worklist("2026-08-24", ["fed"], OK_DOCTOR, SOURCES)
    assert all(j["label"] != "FOMC statement" for j in wl["jobs"])


def test_disabled_source_is_skipped_with_its_probe_reason():
    wl = reach.build_worklist("2026-08-24", ["fed"], OK_DOCTOR, SOURCES)
    dead = [s for s in wl["skipped"] if s["label"] == "Dead source"]
    assert dead and "blocked" in dead[0]["why"]


def test_shared_source_is_fetched_once_and_serves_both_markets():
    wl = reach.build_worklist("2026-08-24", ["hormuz-a", "hormuz-b"], OK_DOCTOR, SOURCES)
    jobs = [j for j in wl["jobs"] if j["label"] == "UKMTO incidents"]
    assert len(jobs) == 1 and sorted(jobs[0]["slugs"]) == ["hormuz-a", "hormuz-b"]


def test_bad_role_or_source_type_in_the_data_file_raises():
    for bad in ({"role": "nonsense"}, {"source_type": "official_everything"}):
        src = {"calendars": {}, "markets": {"m": [{
            "label": "x", "role": "index", "domain": "d", "enabled": True,
            "source_type": "official_primary", "url": "https://d/",
            "trigger": {"kind": "always"}, **bad}]}}
        with pytest.raises(ValueError):
            reach.build_worklist("2026-08-24", ["m"], OK_DOCTOR, src)


def test_worklist_yt_command_actually_writes_a_vtt():
    """--dump-json implies simulate and writes NO subtitle file; --sub-lang en
    (singular) matches nothing. Both were hit live on 2026-08-24."""
    cmd = reach._ytdlp_command("https://www.youtube.com/watch?v=X")
    assert "--print-json" in cmd and "--dump-json" not in cmd
    assert '--sub-langs "en.*"' in cmd


# ----------------------------------------------------------------- ingestion ---

def _write(tmp_path, records) -> tuple:
    p = tmp_path / "reach_items.json"
    p.write_text(json.dumps(records))
    return p, tmp_path


def test_item_not_in_todays_worklist_is_rejected(tmp_path):
    wl = {"jobs": [doc_job()]}
    p, d = _write(tmp_path, [{"id": "not-a-job", "kind": "web", "raw": jina()}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert items == [] and "curated worklist" in rejected[0]["why"]


def test_a_valid_document_becomes_a_displayable_item(tmp_path):
    wl = {"jobs": [doc_job()]}
    p, d = _write(tmp_path, [{"id": "j1", "kind": "web", "raw": jina()}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert rejected == [] and len(items) == 1
    a = items[0]
    assert a["display"] is True                  # headline + source + link
    assert a["reach_source_type"] == "official_primary"
    assert a["reach_slugs"] == ["fed-sep"]
    assert "decided to maintain" in a["scan_text"]


def test_a_transcript_is_never_displayable(tmp_path):
    wl = {"jobs": [video_job()]}
    p, d = _write(tmp_path, [{"id": "v1", "kind": "video",
                              "info": FOMC_INFO, "vtt": vtt()}])
    items, _ = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert items[0]["display"] is False


def test_failures_are_absent_never_placeholders(tmp_path):
    """A naive fetch would have written 'AbuseAlleviationError: … DDoS attack
    suspected' into the packet as evidence text for Stage A to score."""
    wl = {"jobs": [doc_job()]}
    abuse = json.dumps({"code": 403, "message": "AbuseAlleviationError: DDoS "
                                                "attack suspected"})
    p, d = _write(tmp_path, [{"id": "j1", "kind": "web", "raw": abuse}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert items == []
    assert len(rejected) == 1
    assert all("AbuseAlleviation" not in json.dumps(i) for i in items)


def test_discovered_document_must_sit_under_the_curated_prefix(tmp_path):
    parent = {"id": "idx", "role": "index", "label": "CENTCOM releases",
              "domain": "centcom.mil", "source_type": "official_primary",
              "slugs": ["hormuz"],
              "child": {"kind": "document", "immutable": True, "enabled": True,
                        "published_time": "verified",
                        "url_prefix": "https://www.centcom.mil/MEDIA/PUBLIC-RELEASES/Article/"}}
    wl = {"jobs": [parent]}
    good = "https://www.centcom.mil/MEDIA/PUBLIC-RELEASES/Article/1/x/"
    evil = "https://totally-not-centcom.example/Article/1/x/"
    p, d = _write(tmp_path, [
        {"parent": "idx", "kind": "web", "url": good, "raw": jina(url=good)},
        {"parent": "idx", "kind": "web", "url": evil, "raw": jina(url=evil)}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert len(items) == 1 and items[0]["url"] == good
    assert items[0]["domain"] == "centcom.mil"      # inherited, not supplied
    assert "outside the curated prefix" in rejected[0]["why"]


def test_discovered_video_must_come_from_the_curated_channel(tmp_path):
    parent = {"id": "vch", "role": "video", "label": "Fed channel",
              "domain": "youtube.com/@federalreserve",
              "source_type": "official_transcript", "slugs": ["fed"],
              "channel": "@federalreserve",
              "child": {"kind": "video", "channel": "@federalreserve", "enabled": True}}
    wl = {"jobs": [parent]}
    impostor = {**FOMC_INFO, "uploader_id": "@randomreuploader", "channel_id": "UCxxx"}
    p, d = _write(tmp_path, [
        {"parent": "vch", "kind": "video", "info": FOMC_INFO, "vtt": vtt()},
        {"parent": "vch", "kind": "video", "info": impostor, "vtt": vtt()}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert len(items) == 1
    assert "not the curated channel" in rejected[0]["why"]


def test_documents_under_a_disabled_child_are_refused(tmp_path):
    parent = {"id": "idx", "role": "index", "label": "state.gov releases",
              "domain": "state.gov", "source_type": "official_primary", "slugs": ["x"],
              "child": {"kind": "document", "url_prefix": "https://www.state.gov/releases/",
                        "immutable": True, "enabled": False,
                        "why_disabled": "Last-Modified is months stale"}}
    wl = {"jobs": [parent]}
    u = "https://www.state.gov/releases/2026/08/x/"
    p, d = _write(tmp_path, [{"parent": "idx", "kind": "web", "url": u, "raw": jina(url=u)}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert items == [] and "months stale" in rejected[0]["why"]


def test_raw_payload_can_come_from_a_file(tmp_path):
    (tmp_path / "reach_raw").mkdir()
    (tmp_path / "reach_raw" / "j1.md").write_text(jina())
    wl = {"jobs": [doc_job()]}
    p, d = _write(tmp_path, [{"id": "j1", "kind": "web",
                              "raw_file": "reach_raw/j1.md"}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert len(items) == 1 and rejected == []


# ------------------------------------------------------------- source weights --

def test_declared_reach_weights_are_the_approved_values():
    assert sourceweights.OFFICIAL_PRIMARY_W == 1.0
    assert sourceweights.OFFICIAL_STATE_CLAIM_W == 0.5
    assert sourceweights.OFFICIAL_TRANSCRIPT_W == 1.0
    assert sourceweights.UNKNOWN_VIDEO_W == 0.5


def test_procedural_and_state_claim_are_not_collapsed(monkeypatch):
    """'Official' is not 'reliable': a Fed statement about the Fed's own decision
    is procedural ground truth; a state actor's claim about itself is not."""
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "_WEIGHTS", {})
    assert sourceweights.get_weight("federalreserve.gov") == 1.0
    assert sourceweights.get_weight("mod.ru") == 0.5


def test_official_channel_transcript_outweighs_a_random_reuploader(monkeypatch):
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "_WEIGHTS", {})
    assert sourceweights.get_weight("youtube.com/@federalreserve") == 1.0
    assert sourceweights.get_weight("youtube.com/@somebodyelse") == 0.5


def test_blocklist_beats_an_official_reach_domain(monkeypatch):
    """Precedence is unchanged and load-bearing: a blocklisted domain cannot be
    resurrected by a declared row. (Live example: kremlin.ru IS on the Iffy
    index, so it resolves to 0.0 rather than the 0.5 state-claim row.)"""
    monkeypatch.setattr(sourceweights, "_IFFY", {"federalreserve.gov"})
    monkeypatch.setattr(sourceweights, "_WEIGHTS", {})
    assert sourceweights.get_weight("federalreserve.gov") == 0.0


def test_an_uncurated_domain_still_gets_the_unknown_default(monkeypatch):
    monkeypatch.setattr(sourceweights, "_IFFY", set())
    monkeypatch.setattr(sourceweights, "_WEIGHTS", {})
    assert sourceweights.get_weight("some-ministry.example") == sourceweights.UNKNOWN_W


# ------------------------------------------------------------- packet budget ---

def _cfg():
    return {"guardian_q": "federal reserve", "wp_keys": ["federal reserve"],
            "mtype": "slow"}


def _reach_doc(slug="fed", title="FOMC statement", hours_ago=1):
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"title": title, "seendate": when.strftime("%Y%m%dT%H%M%SZ"),
            "domain": "federalreserve.gov", "url": "https://federalreserve.gov/x",
            "trail": "federal reserve statement", "lede": "federal reserve statement",
            "scan_text": "the federal reserve committee decided", "reach": True,
            "reach_kind": "document", "reach_source_type": "official_primary",
            "reach_slugs": [slug], "display": True}


def _packet(monkeypatch, reach_items, n_guardian=8):
    g = [{"title": f"Guardian federal reserve story {i}",
          "seendate": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
          "domain": "theguardian.com", "url": "", "trail": "federal reserve",
          "lede": "federal reserve"} for i in range(n_guardian)]
    monkeypatch.setattr(feeds, "guardian_search", lambda *a, **k: g)
    monkeypatch.setattr(feeds, "wp_day_bullets", lambda d: [])
    monkeypatch.setattr(feeds.time, "sleep", lambda *_: None)
    return feeds.build_packet("fed", _cfg(), reach_items=reach_items)


def test_reach_slot_is_capped_at_two_and_displaces_guardian(monkeypatch):
    items = [_reach_doc(title=f"FOMC statement {i}") for i in range(4)]
    pkt = _packet(monkeypatch, items)
    assert pkt["n_reach"] == reach.MAX_REACH_ITEMS == 2
    assert pkt["guardian_cap"] == reach.GUARDIAN_CAP_WITH_REACH == 4
    n_guardian = sum(1 for a in pkt["articles"] if a["domain"] == "theguardian.com")
    assert n_guardian == 4


def test_guardian_keeps_six_slots_on_a_day_with_no_reach(monkeypatch):
    pkt = _packet(monkeypatch, [])
    assert pkt["n_reach"] == 0 and pkt["guardian_cap"] == 6
    assert sum(1 for a in pkt["articles"] if a["domain"] == "theguardian.com") == 6


def test_a_reach_item_only_reaches_the_market_that_curated_it(monkeypatch):
    """Curated per market: keyword coincidence must not drift an item sideways."""
    pkt = _packet(monkeypatch, [_reach_doc(slug="some-other-market")])
    assert pkt["n_reach"] == 0


def test_official_documents_get_a_fourteen_day_window(monkeypatch):
    """Declared: official documents are state-of-record, not news (14d);
    transcripts age like news (72h)."""
    fresh = _reach_doc(hours_ago=24 * 10)
    stale = _reach_doc(hours_ago=24 * 20)
    assert _packet(monkeypatch, [fresh])["n_reach"] == 1
    assert _packet(monkeypatch, [stale])["n_reach"] == 0


def test_transcripts_get_a_seventytwo_hour_window(monkeypatch):
    t = {**_reach_doc(hours_ago=24 * 5), "reach_kind": "transcript", "display": False}
    assert _packet(monkeypatch, [t])["n_reach"] == 0
    t2 = {**_reach_doc(hours_ago=12), "reach_kind": "transcript", "display": False}
    assert _packet(monkeypatch, [t2])["n_reach"] == 1


# ------------------------------------------------------ page-boundary display --

def _snap_with(article, bd_extra=None):
    from tests.test_newsagent_fv import _snapshot31
    s = _snapshot31()
    s["packet"]["articles"].append(article)
    s["stage_b"]["breakdown"]["articles"].append(
        bd_extra or {"title": article["title"], "domain": article["domain"],
                     "display": article.get("display", True), "c": 0.4,
                     "pp_effect": 4.0})
    return s


def _series():
    from tests.test_newsagent_fv import _series as s
    return s()


TRANSCRIPT_ITEM = {
    "title": "ZQXTRANSCRIPT FOMC Press Conference [2026-07-29 · 90a6dc30]",
    "domain": "youtube.com/@federalreserve", "seendate": "20260729T202431Z",
    "url": "https://www.youtube.com/watch?v=DLFXUkOc_7I",
    "reach": True, "reach_kind": "transcript", "display": False}

DOC_ITEM = {
    "title": "Federal Reserve issues FOMC statement",
    "domain": "federalreserve.gov", "seendate": "20260729T180015Z",
    "url": "https://www.federalreserve.gov/x.htm",
    "reach": True, "reach_kind": "document", "display": True}


def test_a_transcript_is_unreachable_from_the_page_whatever_its_domain():
    sc = dashboard.build_showcase([_snap_with(TRANSCRIPT_ITEM)], _series())
    out = dashboard.render_html(sc)
    assert "ZQXTRANSCRIPT" not in out
    assert "youtube.com/@" not in out
    assert "watch?v=DLFXUkOc_7I" not in out


def test_a_transcript_increments_the_private_count():
    sc = dashboard.build_showcase([_snap_with(TRANSCRIPT_ITEM)], _series())
    card = sc["markets"][0]
    assert card["n_private_items"] == 1 and card["n_transcripts"] == 1
    assert all("ZQXTRANSCRIPT" not in e["title"] for e in card["evidence"])


def test_a_transcript_never_reaches_the_shared_evidence_feed():
    sc = dashboard.build_showcase([_snap_with(TRANSCRIPT_ITEM)], _series())
    assert all("ZQXTRANSCRIPT" not in i["title"] for i in sc["feed"]["items"])
    assert sc["feed"]["n_private"] >= 1


def test_an_official_document_shows_headline_source_and_link():
    """The pdf_ingest treatment, APPROVED 2026-08-24: headline + source + link on
    the public page, body internal-only."""
    sc = dashboard.build_showcase([_snap_with(DOC_ITEM)], _series())
    card = sc["markets"][0]
    ev = [e for e in card["evidence"] if e["domain"] == "federalreserve.gov"]
    assert len(ev) == 1
    assert ev[0]["title"] == "Federal Reserve issues FOMC statement"
    assert ev[0]["url"] == "https://www.federalreserve.gov/x.htm"
    assert card["n_reach_docs"] == 1
    out = dashboard.render_html(sc)
    assert "Federal Reserve issues FOMC statement" in out


def test_a_document_body_is_never_rendered():
    doc = {**DOC_ITEM, "scan_text": "ZQXBODY the committee decided to maintain",
           "lede": "ZQXBODY the committee decided to maintain"}
    out = dashboard.render_html(dashboard.build_showcase([_snap_with(doc)], _series()))
    assert "ZQXBODY" not in out


def test_the_ledger_driver_line_never_carries_a_transcript_title():
    """`drivers` is public copy AND is written to the append-only ledger."""
    bd = {"title": TRANSCRIPT_ITEM["title"], "domain": TRANSCRIPT_ITEM["domain"],
          "display": False, "c": 0.4, "pp_effect": 4.0}
    label = (fvmodel._source_label(bd["domain"]) + " — private analysis item") \
        if not bd.get("display", True) else bd["title"]
    assert "ZQXTRANSCRIPT" not in label and "transcript" in label


def test_ip_scrub_still_clean_with_reach_items_present():
    sc = dashboard.build_showcase(
        [_snap_with(TRANSCRIPT_ITEM), _snap_with(DOC_ITEM)], _series())
    out = dashboard.render_html(sc)
    for secret in ("ZQXTRANSCRIPT", "alpha=", "fv_state", "feature_cache"):
        assert secret not in out


def test_breakdown_row_without_a_display_flag_still_hides_a_transcript():
    """Belt and braces: this boundary must not depend on an upstream field."""
    legacy = {"title": "ZQXTRANSCRIPT leaked", "domain": "youtube.com/@federalreserve",
              "c": 0.4, "pp_effect": 4.0}          # no display key at all
    dom, title = dashboard._bd_row_public(legacy)
    assert "ZQXTRANSCRIPT" not in title and "not displayed" in title


# ---------------------------------------------------------- curated data file --

def test_the_shipped_source_file_is_well_formed():
    src = reach.load_sources()
    for slug, entries in src["markets"].items():
        for e in entries:
            assert e["role"] in reach.ROLES, (slug, e["label"])
            assert e["source_type"] in reach.SOURCE_TYPES, (slug, e["label"])
            assert e.get("why"), (slug, e["label"])
            assert e.get("probe", {}).get("date"), (slug, e["label"])
            if e["role"] != "video":
                assert e["url"].startswith("http"), (slug, e["label"])


def test_every_curated_market_is_on_the_live_slate():
    from newsagent import config
    src = reach.load_sources()
    unknown = set(src["markets"]) - set(config.LIVE_MARKETS)
    assert not unknown, f"reach_sources references retired/unknown slugs: {unknown}"
    documented = set(src["markets"]) | set(src["markets_deliberately_without_sources"])
    assert set(config.LIVE_MARKETS) - documented == set(), \
        "every live market must either have curated sources or a recorded reason"


def test_state_actor_domains_are_not_curated_as_procedural():
    """The propaganda guard, enforced against the DATA file, not just the code."""
    src = reach.load_sources()
    for slug, entries in src["markets"].items():
        for e in entries:
            if e["domain"] in sourceweights.OFFICIAL_STATE_CLAIM_DOMAINS:
                assert e["source_type"] == "official_state_claim", (slug, e["label"])


def test_a_fetch_clock_timestamp_is_refused():
    """MEASURED LIVE 2026-08-24: centcom.mil served `Published Time: Mon, 24 Aug
    2026 13:28:03 GMT` for an article datelined July 29, 2026 — 26 days of drift,
    in the direction that makes a stale document look like breaking news. It is
    never AFTER the cutoff, so the approved after-cutoff check cannot catch it."""
    job = doc_job(domain="centcom.mil", timestamp_class="fetch_clock",
                  timestamp_trusted=False)
    item, why = reach.item_from_jina(job, jina(published="Mon, 24 Aug 2026 13:28:03 GMT"),
                                     CUTOFF)
    assert item is None and "fetch clock" in why


def test_an_unmeasured_source_cannot_date_evidence():
    job = doc_job(timestamp_class=None, timestamp_trusted=None)
    item, why = reach.item_from_jina(job, jina(), CUTOFF)
    assert item is None and "not a verified publication instant" in why


def test_only_verified_sources_carry_a_trusted_timestamp_in_the_worklist():
    src = {"calendars": {}, "markets": {"m": [
        {"label": "trusted", "role": "document", "domain": "federalreserve.gov",
         "source_type": "official_primary", "immutable": True, "enabled": True,
         "url": "https://federalreserve.gov/a.htm", "trigger": {"kind": "always"},
         "probe": {"date": "2026-08-24", "published_time": "verified"}},
        {"label": "untrusted", "role": "document", "domain": "centcom.mil",
         "source_type": "official_primary", "immutable": True, "enabled": True,
         "url": "https://centcom.mil/a", "trigger": {"kind": "always"},
         "probe": {"date": "2026-08-24", "published_time": "fetch_clock"}}]}}
    wl = reach.build_worklist("2026-08-24", ["m"], OK_DOCTOR, src)
    by_label = {j["label"]: j for j in wl["jobs"]}
    assert by_label["trusted"]["timestamp_trusted"] is True
    assert by_label["untrusted"]["timestamp_trusted"] is False


def test_the_data_file_never_claims_verified_without_a_probe():
    """A config edit must not be able to promote a source into the trusted class
    without a recorded measurement behind it."""
    src = reach.load_sources()
    for slug, entries in src["markets"].items():
        for e in entries:
            cls = (e.get("probe") or {}).get("published_time")
            assert cls in reach.TIMESTAMP_CLASSES, (slug, e["label"], cls)


# ------------------------------------ DL-1…DL-6: the two-source dateline rule ---
# Amendment locked 2026-08-24 (Justin's variant: tie-break, not discard).

STATE_DL = {
    "url_pattern": r"/(?P<year>20\d{2})/(?P<month>\d{1,2})/",
    "body_pattern": (r"(?P<month_name>January|February|March|April|May|June|July|"
                     r"August|September|October|November|December)\s+"
                     r"(?P<day>\d{1,2}),\s+(?P<year>20\d{2})"),
}
STATE_URL = "https://www.state.gov/releases/office-of-the-spokesperson/2026/08/x/"


def dl_body(dateline="August 20, 2026") -> str:
    """A state.gov release in the shape the chrome-strip actually leaves behind:
    headline, kicker, dateline, then multi-paragraph prose. Long enough to clear
    MIN_DOC_CHARS, because the prose floor runs before the dateline path does."""
    return (f"U.S. Sanctions Smuggling Network for Qods Force and Hizballah\n"
            f"Press Statement\n{dateline}\n"
            "For decades the Iranian regime has used proxies to spread instability "
            "throughout the Middle East, and today the Department is renewing its "
            "designations against that network in response.\n"
            "The designations announced today target individuals and entities that "
            "have facilitated the movement of funds and materiel on behalf of the "
            "network, including shipping agents and front companies operating "
            "across several jurisdictions.\n"
            "The Department will continue to use every tool available to disrupt "
            "these financing channels, and urges partners in the region to take "
            "corresponding steps within their own jurisdictions.\n")


def dl_job(**over) -> dict:
    job = doc_job(domain="state.gov", url=STATE_URL, timestamp_class="absent",
                  timestamp_trusted=False, dateline=STATE_DL)
    job.update(over)
    return job


def dl_raw(dateline="August 20, 2026", published=None) -> str:
    return jina(title="U.S. Sanctions Smuggling Network", published=published,
                url=STATE_URL, body=dl_body(dateline))


# --- DL-2: the two legs ---------------------------------------------------

def test_url_leg_month_precision_yields_the_whole_month():
    assert reach.parse_url_date(STATE_URL, STATE_DL["url_pattern"]) == \
        (date(2026, 8, 1), date(2026, 8, 31))


def test_url_leg_day_precision_collapses_to_that_day():
    pat = r"/(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})/"
    assert reach.parse_url_date("https://x.gov/2026/08/20/slug/", pat) == \
        (date(2026, 8, 20), date(2026, 8, 20))


def test_url_leg_absent_is_none():
    assert reach.parse_url_date("https://www.centcom.mil/MEDIA/Article/4559495/x/",
                                STATE_DL["url_pattern"]) is None


def test_body_leg_takes_the_first_match_not_a_date_in_the_prose():
    body = dl_body("August 20, 2026") + "\nThe scheme ran from March 3, 2021 onward.\n"
    assert reach.parse_body_dateline(body, STATE_DL["body_pattern"]) == date(2026, 8, 20)


# --- DL-3: agreement ------------------------------------------------------

def test_agreement_dates_the_item_at_the_body_day_end_of_day():
    dt, conflict, why = reach.resolve_dateline(STATE_DL, STATE_URL, dl_body())
    assert why == "" and conflict is False
    assert dt == datetime(2026, 8, 20, 23, 59, 59, tzinfo=timezone.utc)


# --- DL-4: the tie-break, in BOTH directions ------------------------------

def test_conflict_in_a_live_packet_takes_the_earlier_bound():
    """A stale document must not present as fresh — the centcom.mil failure mode."""
    dt, conflict, why = reach.resolve_dateline(
        STATE_DL, STATE_URL, dl_body("March 4, 2026"), reach.MODE_LIVE)
    assert conflict is True and why == ""
    assert dt == datetime(2026, 3, 4, 23, 59, 59, tzinfo=timezone.utc)


def test_conflict_in_reconstruction_takes_the_later_bound():
    """Nothing may enter a packet dated before it existed."""
    dt, conflict, _ = reach.resolve_dateline(
        STATE_DL, STATE_URL, dl_body("March 4, 2026"), reach.MODE_RECONSTRUCTION)
    assert conflict is True
    assert dt == datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_the_tie_break_is_never_a_discard():
    for mode in (reach.MODE_LIVE, reach.MODE_RECONSTRUCTION):
        dt, _, why = reach.resolve_dateline(
            STATE_DL, STATE_URL, dl_body("January 2, 2026"), mode)
        assert dt is not None and why == ""


def test_a_conflict_is_flagged_on_the_item_so_the_rate_is_countable():
    item, why = reach.item_from_jina(dl_job(), dl_raw("March 4, 2026"), CUTOFF)
    assert why == ""
    assert item["dateline_conflict"] is True
    assert item["dated_by"] == "dateline_conflict"
    assert item["seendate"] == "20260304T235959Z"


def test_agreement_is_not_flagged_as_a_conflict():
    item, _ = reach.item_from_jina(dl_job(), dl_raw(), CUTOFF)
    assert item["dateline_conflict"] is False and item["dated_by"] == "dateline"


# --- DL-5: either leg missing --------------------------------------------

def test_a_body_dateline_alone_is_not_enough():
    """centcom.mil: real dateline in the body, no date anywhere in the URL."""
    job = dl_job(url="https://www.centcom.mil/MEDIA/PUBLIC-RELEASES/Article/4559495/x/")
    raw = jina(title="U.S. Strikes IRGC Targets", published=None,
               url=job["url"], body=dl_body("July 29, 2026"))
    item, why = reach.item_from_jina(job, raw, CUTOFF)
    assert item is None and "structural leg is missing" in why


def test_a_url_date_alone_is_not_enough():
    raw = jina(title="No dateline here", published=None, url=STATE_URL,
               body="This release carries no parseable dateline anywhere in its body "
                    "text, only prose about the designations announced today and the "
                    "steps partners are urged to take.\n"
                    "The measures build on previous actions and are intended to "
                    "disrupt the financing channels that sustain the network across "
                    "several jurisdictions.\n"
                    "The Department will continue to use every tool available and "
                    "will provide further detail through the usual channels in due "
                    "course as the review proceeds.\n")
    item, why = reach.item_from_jina(dl_job(), raw, CUTOFF)
    assert item is None and "content leg is missing" in why


def test_neither_leg_is_reported_as_single_source_dating():
    dt, _, why = reach.resolve_dateline(STATE_DL, "https://x.gov/no-date/", "no dates")
    assert dt is None and "single-source dating is not permitted" in why


# --- DL-6: the hard refusal, tested adversarially ------------------------

def test_a_future_dated_document_is_refused_outright():
    """The one thing a CONTENT-derived date can do that an infrastructure one
    cannot: invent a document from the future."""
    url = "https://www.state.gov/releases/office-of-the-spokesperson/2027/06/x/"
    job = dl_job(url=url)
    raw = jina(title="From the future", published=None, url=url,
               body=dl_body("June 14, 2027"))
    item, why = reach.item_from_jina(job, raw, CUTOFF)
    assert item is None and "future-dated" in why


def test_a_future_dateline_cannot_sneak_through_the_conflict_tie_break():
    """Adversarial: agreeing legs are the obvious case. A CONFLICT whose
    reconstruction branch resolves into the future must die too."""
    url = "https://www.state.gov/releases/office-of-the-spokesperson/2027/06/x/"
    job = dl_job(url=url)
    raw = jina(title="Conflicted and future", published=None, url=url,
               body=dl_body("March 4, 2026"))
    item, why = reach.item_from_jina(job, raw, CUTOFF,
                                     mode=reach.MODE_RECONSTRUCTION)
    assert item is None and "future-dated" in why


def test_the_fetch_instant_narrows_the_horizon_below_the_cutoff():
    """A record may supply its own fetched_at; the tighter bound wins."""
    job = dl_job()
    raw = dl_raw("August 20, 2026")
    early = datetime(2026, 8, 1, tzinfo=timezone.utc)
    item, why = reach.item_from_jina(job, raw, CUTOFF, fetch_instant=early)
    assert item is None and "future-dated" in why
    ok, _ = reach.item_from_jina(job, raw, CUTOFF)
    assert ok is not None


# --- DL-1: sources without a dateline block are untouched -----------------

def test_a_source_without_a_dateline_block_follows_the_old_rule_unchanged():
    item, why = reach.item_from_jina(
        doc_job(domain="centcom.mil", timestamp_class="fetch_clock",
                timestamp_trusted=False),
        jina(published="Mon, 24 Aug 2026 13:28:03 GMT"), CUTOFF)
    assert item is None and "fetch clock" in why


def test_a_verified_source_still_uses_its_published_time_header():
    """The primary path wins wherever it works; the amendment never overrides it."""
    job = doc_job(dateline=STATE_DL)      # verified AND carrying a dateline block
    item, why = reach.item_from_jina(job, jina(), CUTOFF)
    assert why == "" and item["seendate"] == "20260729T180015Z"
    assert item["dated_by"] == "published_time"


def test_a_verified_source_falls_back_when_its_header_is_missing_on_a_page():
    job = doc_job(domain="state.gov", url=STATE_URL, dateline=STATE_DL)
    item, why = reach.item_from_jina(job, dl_raw(), CUTOFF)
    assert why == "" and item["dated_by"] == "dateline"


def test_an_undated_source_with_no_block_still_says_undated():
    item, why = reach.item_from_jina(doc_job(), jina(published=None), CUTOFF)
    assert item is None and "Published Time" in why


# --- provenance: a discovered child inherits the curated spec -------------

def test_a_discovered_child_inherits_its_parents_dateline_spec(tmp_path):
    parent = {"id": "idx", "role": "index", "label": "state.gov releases",
              "domain": "state.gov", "source_type": "official_primary",
              "slugs": ["iran"], "timestamp_class": "absent",
              "child": {"kind": "document", "enabled": True, "immutable": True,
                        "url_prefix": "https://www.state.gov/releases/",
                        "dateline": STATE_DL}}
    wl = {"jobs": [parent]}
    p, d = _write(tmp_path, [{"parent": "idx", "kind": "web", "url": STATE_URL,
                              "raw": dl_raw()}])
    items, rejected = reach.ingest_reach_file(p, wl, d, CUTOFF)
    assert rejected == [] and len(items) == 1
    assert items[0]["seendate"] == "20260820T235959Z"
    assert items[0]["domain"] == "state.gov"


# --- the shipped data file ------------------------------------------------

def test_every_declared_dateline_block_has_both_patterns_and_a_probe():
    src = reach.load_sources()
    for slug, entries in src["markets"].items():
        for e in entries:
            for spec in (e.get("dateline"), (e.get("child") or {}).get("dateline")):
                if not spec:
                    continue
                assert spec.get("url_pattern") and spec.get("body_pattern"), \
                    (slug, e["label"])
                assert spec.get("probe", {}).get("date"), (slug, e["label"])
                import re as _re
                _re.compile(spec["url_pattern"])
                _re.compile(spec["body_pattern"])


def test_a_child_is_only_enabled_with_a_trusted_header_or_a_dateline_block():
    """The amendment is the ONLY thing that may enable an untrusted source."""
    src = reach.load_sources()
    for slug, entries in src["markets"].items():
        for e in entries:
            c = e.get("child") or {}
            if c.get("kind") != "document" or not c.get("enabled"):
                continue
            trusted = c.get("published_time",
                            (e.get("probe") or {}).get("published_time")) == "verified"
            assert trusted or c.get("dateline"), (slug, e["label"])
