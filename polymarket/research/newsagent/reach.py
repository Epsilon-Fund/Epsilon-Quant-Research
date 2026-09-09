"""agent-reach evidence extension — primary-source reach for the observatory (v3.4).

Scope is exactly what [[newsagent_agentreach_scoping]] signed off on 2026-08-24:
**Jina Reader** for curated official resolution-source documents (ADOPT, narrow)
and **event-triggered YouTube transcripts** (ADOPT-NARROW). Exa stays a discovery
aid that never becomes an evidence row, and every social/cookie channel is out of
scope. Nothing here fetches on its own — the attended session shells out with the
agent-reach skill and this module supplies the worklist and the gate.

The idiom is the THIRD instance of the existing out-of-band pattern (after
`--stage onboard --priors-file` and `--stage extract --features-file`):

  1. `--stage fetch` writes `reach_worklist.json` into the day dir: which URL /
     video to fetch for which market today, why, and the exact shell command.
  2. The attended agent runs those commands and writes the RAW payloads back as
     `reach_items.json` (inline or as files under `reach_raw/`).
  3. `--stage fetch --reach-file reach_items.json` runs the gate here and merges
     the survivors into the packets through the existing `_keyword_filter` ->
     `build_packet` path.

**The gate lives in this repo, not in the agent.** The agent hands over bytes; the
validation, the timestamps, the display flags and the weights are ours, because
agent-reach supplies none of them (it has no cache, no retry and no persistence of
any kind). A failed item is ABSENT — never a placeholder, never an error string
stored as evidence text. That rule is not decoration: the scoping probe caught
`r.jina.ai` returning an `AbuseAlleviationError` JSON body for reuters.com and a
state.gov **404 page served as HTTP 200**, either of which a naive fetch would have
written into the packet as evidence for Stage A to extract features from.

Timestamp discipline (verbatim from the scoping note § Timestamp discipline, which
is the same rule the GDELT client-side seendate filter enforces):

  * **Jina page** — the item is dated by the `Published Time:` header, which is
    empirically the origin's HTTP `Last-Modified`. That equals publication for an
    IMMUTABLE DATED DOCUMENT and is merely a touch-time for a ROLLING page, so a
    rolling page can never become a dated evidence item: it is `role="index"` in
    the worklist, fetched only so the attended agent can find today's dated
    document, and `item_from_jina` refuses to build an item from one.
  * **Transcript** — dated at `max(upload, release_timestamp + duration)`, i.e.
    the later of upload time and stream end, because a transcript covers the whole
    event and is only complete when the stream ends. The scoping probe measured a
    third-party FOMC re-stream whose `release_timestamp` was 17:57:02Z — three
    minutes BEFORE the 18:00:15Z statement it contained.
  * Anything after the run cutoff is dropped, in both channels.

Windows (declared, not fitted): official documents are state-of-record rather than
news and carry 14 days; transcripts carry 72h like the news window.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import DATA

# --------------------------------------------------------------------------
# Declared constants (judgments, not fits — n is far too small to fit any of
# them, same posture as the RSP tiers and the lean multipliers).
# --------------------------------------------------------------------------
JINA_PREFIX = "https://r.jina.ai/"
DOCTOR_TIMEOUT_S = 5           # doctor is a cheap pre-flight, never a blocker
OFFICIAL_DOC_WINDOW_DAYS = 14  # official documents are state-of-record, not news
TRANSCRIPT_WINDOW_HOURS = 72   # transcripts age like news
MAX_REACH_ITEMS = 2            # packet slot cap; displaces Guardian 6 -> 4
GUARDIAN_CAP_WITH_REACH = 4    # approved 2026-08-24 (sign-off block 1)
GUARDIAN_CAP_DEFAULT = 6
MIN_DOC_CHARS = 400            # a real document after chrome-strip
MIN_TRANSCRIPT_WORDS = 200     # an empty caption response is a known transient
SCAN_CHARS = 4000              # internal keyword-match surface cap (pdf_ingest parity)
NAV_ITEM_MAX = 120             # linked list entries shorter than this are nav chrome
LINK_LINE_MIN = 40             # a bare linked line shorter than this is chrome ([PDF])
PROSE_WORDS_MIN = 12           # a line this long anchors a content block
ANCHOR_GAP = 1                 # short lines this close to an anchor are kept

SOURCES_FILE = Path(__file__).with_name("reach_sources.json")

# Timestamp classes a curated source can be measured into (probe.published_time).
# Only `verified` may produce a dated evidence item. This is a code-enforced
# TIGHTENING of the approved rule, added during the build after a live measurement
# on 2026-08-24 showed the approved rule alone is not sufficient:
#
#   verified     the header equals the document's release instant. Measured true
#                on federalreserve.gov static press releases, and (so far) nowhere
#                else in the curated set.
#   touch        a real edit time on a rolling page (fomccalendars.htm,
#                sos.ca.gov). Already handled: such a page is role="index".
#   fetch_clock  THE DANGEROUS ONE. The stamp tracks the FETCH instant, so it is
#                always "now". centcom.mil served `Published Time: Mon, 24 Aug
#                2026 13:28:03 GMT` for an article datelined July 29, 2026 —
#                26 days of drift, in the direction that makes stale content look
#                like breaking news. It is never after the cutoff, so the
#                approved after-cutoff check cannot catch it, and it would have
#                entered a month-old document as today's evidence.
#   absent       no header at all (state.gov, congress.gov, ukmto.org, nato.int,
#                imo.org, war.gov, whitehouse.gov, tse.jus.br, gov.il,
#                conseil-constitutionnel.fr). Safe: the approved rule discards it.
#   blocked      r.jina.ai could not reach the origin (403/404).
#
# Why this is a tightening and not a design change: it can only ever make an item
# ABSENT. It never admits anything the approved rule would have rejected, and it
# introduces no new timestamp source. Loosening the rule — e.g. dating a document
# by the dateline printed in its body — would be a design change and is NOT done
# here; it is written up as a proposed amendment for sign-off.
TIMESTAMP_CLASSES = {"verified", "touch", "fetch_clock", "absent", "blocked"}
TRUSTED_TIMESTAMP = "verified"

# Source types that may be declared in reach_sources.json. The weight each one
# carries lives in sourceweights.py (approved 2026-08-24); this set only guards
# against a typo in the data file silently becoming an unknown-weight source.
SOURCE_TYPES = {"official_primary", "official_state_claim",
                "official_transcript", "unknown_video"}
ROLES = {"document", "index", "video"}


# --------------------------------------------------------------------------
# Pre-flight: the doctor gate (channel level)
# --------------------------------------------------------------------------
def doctor(day_dir: Path | None = None, force: bool = False,
           timeout: int = DOCTOR_TIMEOUT_S) -> dict:
    """`agent-reach doctor --json`, cached into the day dir as reach_doctor.json.

    Mirrors `_refresh_gdelt` exactly: this can only ever DEGRADE the run. A
    machine that never installed agent-reach gets `available: False` and the
    pipeline runs identically to before this module existed.
    """
    cache = (day_dir / "reach_doctor.json") if day_dir else None
    if cache is not None and cache.exists() and not force:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    binary = shutil.which("agent-reach")
    if binary is None:
        rec = {"available": False, "why": "agent-reach is not installed on this "
               "machine (uv tool install …/agent-reach)", "channels": {}}
    else:
        try:
            proc = subprocess.run([binary, "doctor", "--json"], capture_output=True,
                                  text=True, timeout=timeout)
            rec = {"available": True, "why": "", "channels": json.loads(proc.stdout)}
        except subprocess.TimeoutExpired:
            rec = {"available": False,
                   "why": f"agent-reach doctor timed out after {timeout}s", "channels": {}}
        except Exception as e:  # unparseable output, non-zero exit, anything
            rec = {"available": False, "why": f"agent-reach doctor failed ({e})",
                   "channels": {}}
    rec["checked_at"] = datetime.now(timezone.utc).isoformat()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec, indent=1))
    return rec


def channel_status(doc: dict, channel: str) -> tuple[bool, str]:
    """(ok, why-not) for one agent-reach channel.

    `doctor` is a NECESSARY gate, not a sufficient one — the scoping pass proved
    green-at-the-channel says nothing about the specific URL you are about to
    fetch (`web` was `ok` while r.jina.ai refused reuters.com outright). The
    per-item gate below is what actually decides whether an object is evidence.
    """
    if not doc.get("available"):
        return False, doc.get("why", "agent-reach unavailable")
    ch = (doc.get("channels") or {}).get(channel)
    if ch is None:
        return False, f"channel {channel!r} not reported by doctor"
    if ch.get("status") != "ok":
        # doctor's prose message is localized by the CLI's LANG; the status and
        # backend fields are not, so the gate reads those and quotes the message.
        return False, f"{ch.get('status')} — {(ch.get('message') or '').splitlines()[0][:160]}"
    return True, ""


# --------------------------------------------------------------------------
# Chrome strip (mandatory — 4.3% of the FOMC page was the statement)
# --------------------------------------------------------------------------
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LIST_RE = re.compile(r"^\s*(?:[*+\-]|\d+\.)\s+")
_NAV_RE = re.compile(
    r"(?i)^\s*(skip to (the )?(main )?content|expand sub-?menu|collapse sub-?menu|"
    r"share|print|email|back to top|top of page|subscribe|sign ?up|sign ?in|log ?in|"
    r"follow us|cookie[s]?( settings| policy)?|accept( all)?|reject( all)?|menu|"
    r"search|toggle navigation|breadcrumb|skip navigation|stay connected|"
    r"[\w ]*toggle (button|menu|dropdown)[\w ]*|toggle dropdown menu|"
    r"(search|submit|button|toggle|menu|close|open|next|previous|home|more|"
    r"navigation)( +(search|submit|button|toggle|menu|close|open|next|previous|"
    r"home|more|navigation))*)\s*[:.]?\s*$")
LINK_RUN_MIN = 3      # this many links on one line...
LINK_RUN_LABEL_MAX = 40   # ...with labels this short is a link bar, not prose

# Site boilerplate that survives every other rule because it IS prose — the US
# government .gov interstitial is the worst offender and appears on every federal
# page we fetch. Same idea as pdf_ingest.DISCLAIMER_PATTERNS: a declared line-level
# blocklist, applied before Stage A ever sees the text.
BOILERPLATE_PATTERNS = [
    r"(?i)^an official website of the (united states|u\.s\.) government",
    r"(?i)^official websites use \.gov",
    r"(?i)^a \*{0,2}\.gov\*{0,2} website belongs to an official government",
    r"(?i)^secure \.gov websites use https",
    r"(?i)^a \*{0,2}lock\*{0,2} \(.{0,20}\) or \*{0,2}https",
    r"(?i)share sensitive information only on official, secure websites",
    r"(?i)^back to home",
    r"(?i)^(this|the) (site|website) uses cookies",
    r"(?i)^javascript (is|must be) (disabled|enabled)",
    r"(?i)^your browser (is out of date|does not support)",
    # Cookie/consent banners. These are PROSE, so the prose-anchor pass keeps
    # them; state.gov puts ~3KB of consent text above its own dateline. Declared
    # line blocklist, same treatment as the .gov interstitial.
    r"(?i)^we use cookies to",
    r"(?i)^the technical storage or access",
    r"(?i)^(functional|preferences|statistics|marketing)-? ?(\[x\]|always active)",
    r"(?i)^(manage|view|save) (consent|preferences|options)",
    r"(?i)without a subpoena, voluntary compliance",
    r"(?i)^by continuing to use (the|this) site",
    r"(?i)^(accept|decline|reject) (all )?cookies",
    # Site-wide alert bars. state.gov runs a global travel/consular banner above
    # every release; it is prose, it survives every other rule, and it is the
    # first thing prose_paragraphs would otherwise hand Stage A as the lede.
    r"(?i)needing consular assistance can call",
    r"(?i)^enroll in step",
    r"(?i)^latest alerts\s*$",
]
_BOILER_RE = [re.compile(p) for p in BOILERPLATE_PATTERNS]
_HEADING_RE = re.compile(r"^\s*#{1,4}\s+\S")   # h5/h6 are footer furniture


def _delink(line: str) -> str:
    """Drop images entirely (alt text is never document prose) and reduce links to
    their anchor text."""
    return _LINK_RE.sub(r"\1", _IMG_RE.sub("", line))


def strip_chrome(markdown: str, max_chars: int = SCAN_CHARS) -> str:
    """Jina markdown -> the document's prose, with site navigation removed.

    Mandatory, not cosmetic: the scoping probe measured the July FOMC statement at
    **52 KB of markdown of which 4.3% was the statement**, so an un-stripped page
    costs ~13k tokens of navigation junk for Stage A to extract features from. The
    `pdf_ingest` disclaimer-strip is the precedent for doing this before Stage A.

    Two passes, both declared:

      1. **Line chrome** — drop link-only/image-only lines, known nav phrases,
         declared site boilerplate (the .gov interstitial), linked list entries
         with little text, and bare linked fragments (`[PDF](…)`).
      2. **Prose anchoring** — site menus survive pass 1 as long runs of short
         bolded list items, so a line is kept only if it is a heading, is itself
         prose (>= PROSE_WORDS_MIN words), or sits within ANCHOR_GAP lines of one
         that is. That keeps a dateline or a "For release at 2:00 p.m. EDT" next
         to its paragraph and drops a hundred-entry menu that touches no prose.
    """
    body = markdown or ""
    m = re.search(r"^Markdown Content:\s*$", body, re.M)
    if m:
        body = body[m.end():]
    kept: list[str] = []
    for raw in body.splitlines():
        if not raw.strip():
            continue
        txt = re.sub(r"\s+", " ", _delink(raw)).strip()
        if not txt:                       # link-only or image-only line
            continue
        if _NAV_RE.match(txt):
            continue
        stripped_marks = txt.lstrip("*+-# ").strip()
        if any(rx.search(stripped_marks) for rx in _BOILER_RE):
            continue
        n_links = len(_LINK_RE.findall(_IMG_RE.sub("", raw)))
        if n_links >= LINK_RUN_MIN and len(txt) / n_links < LINK_RUN_LABEL_MAX:
            continue                      # a run of short link labels = a link bar
        has_link = n_links > 0
        if has_link and _LIST_RE.match(raw) and len(txt) < NAV_ITEM_MAX:
            continue                      # site-navigation list entry
        if has_link and len(txt) < LINK_LINE_MIN:
            continue                      # bare linked fragment
        if kept and kept[-1][1] == txt:
            continue                      # site furniture repeated verbatim
        kept.append((raw, txt))

    anchors = [i for i, (raw, txt) in enumerate(kept)
               if _HEADING_RE.match(raw)
               or len(re.sub(r"[*_`#>]", "", txt).split()) >= PROSE_WORDS_MIN]
    if not anchors:
        return ""
    keep_idx = set()
    for a in anchors:
        for j in range(max(0, a - ANCHOR_GAP), min(len(kept), a + ANCHOR_GAP + 1)):
            keep_idx.add(j)
    out: list[str] = []
    prev = None
    for i in sorted(keep_idx):
        if prev is not None and i > prev + 1:
            out.append("")                # a dropped run becomes a paragraph break
        out.append(re.sub(r"^\s*(?:[*+\-]|\d+\.)\s+", "", kept[i][1]).strip())
        prev = i
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text[:max_chars]


def strip_vtt(vtt: str) -> str:
    """WebVTT -> plain transcript text (cue timings, indices and markup removed).

    Consecutive duplicate lines are collapsed: auto-caption tracks repeat each
    line as the rolling window advances, which would otherwise triple the word
    count and let an empty-ish track clear the 200-word floor.
    """
    lines: list[str] = []
    for raw in (vtt or "").splitlines():
        s = raw.strip()
        if not s or "-->" in s:
            continue
        if s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")):
            continue
        if re.fullmatch(r"\d+", s):
            continue
        s = re.sub(r"<[^>]+>", "", s)
        s = s.replace("&nbsp;", " ").replace("&amp;", "&")
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        if lines and lines[-1] == s:
            continue
        lines.append(s)
    return " ".join(lines).strip()


# --------------------------------------------------------------------------
# Per-item validation (the part doctor cannot do)
# --------------------------------------------------------------------------
_TS_FORMATS = ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
               "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S")


def parse_published(text: str) -> datetime | None:
    """`Published Time:` value -> aware UTC datetime (None when unparseable)."""
    t = (text or "").strip()
    if not t:
        return None
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(t, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None \
                else dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None \
            else dt.astimezone(timezone.utc)
    except ValueError:
        return None


def parse_jina(raw: str) -> tuple[dict | None, str]:
    """Validate a raw r.jina.ai response. Returns (parsed, reason-it-failed).

    The four checks are the ones the scoping probe established are all necessary,
    in the order that makes each failure legible. Two of the three observed
    failure modes returned HTTP 200 or a well-formed body, so "the curl
    succeeded" is worthless as a success test:

      1. a JSON error envelope (`{"code": 403, …}`) — reuters.com anonymous block;
      2. `Warning: Target URL returned error 404` — state.gov 404-served-as-200;
      3. a missing `Title:` header — not a rendered page at all;
      4. too little prose after the chrome-strip — a stub, an interstitial, or a
         page whose body did not render.
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty response"
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            code = obj.get("code")
            if isinstance(code, str) and code.isdigit():
                code = int(code)
            msg = str(obj.get("message") or obj.get("readableMessage")
                      or obj.get("name") or "")[:200]
            if isinstance(code, int) and code >= 400:
                return None, f"jina returned error code {code}: {msg}"
            return None, f"jina returned a JSON body, not a page: {msg or text[:120]}"
    head = text[:4000]
    m = re.search(r"Warning: Target URL returned error (\d+)[^\n]*", head)
    if m:
        return None, f"origin returned {m.group(1)} (served inside a 200 response)"
    title_m = re.search(r"^Title:\s*(.+?)\s*$", head, re.M)
    if not title_m:
        return None, "no Title: header in the jina response"
    title = title_m.group(1).strip()
    if not title:
        return None, "empty Title: header"
    url_m = re.search(r"^URL Source:\s*(\S+)\s*$", head, re.M)
    pub_m = re.search(r"^Published Time:\s*(.+?)\s*$", head, re.M)
    body = strip_chrome(text)
    if len(body) < MIN_DOC_CHARS:
        return None, (f"only {len(body)} chars of prose after chrome-strip "
                      f"(floor {MIN_DOC_CHARS})")
    return ({"title": title, "url": url_m.group(1) if url_m else "",
             "published_raw": pub_m.group(1) if pub_m else "",
             "published": parse_published(pub_m.group(1) if pub_m else ""),
             "body": body}, "")


def _compact(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def content_unique_title(title: str, dt: datetime, body: str, immutable: bool) -> str:
    """Stage-A cache keys are `sha1(slug|title|vN)`, so a title that repeats with
    DIFFERENT content silently returns yesterday's features for today's text.

    An immutable dated document is unique per event by construction and keeps its
    title verbatim (which matters — official documents display their headline).
    Anything else gets a date + content-hash suffix so the cache key moves when
    the content moves. Those items are `display=False` by construction, so the
    suffix is never public.
    """
    if immutable:
        return title
    h = hashlib.sha1((body or "").encode()).hexdigest()[:8]
    return f"{title} [{dt.astimezone(timezone.utc):%Y-%m-%d} · {h}]"


# --------------------------------------------------------------------------
# DL-1 … DL-6: the two-source dateline path (AMENDMENT, locked 2026-08-24)
# --------------------------------------------------------------------------
# A LOOSENING of the approved timestamp discipline, adopted by Justin on
# 2026-08-24 in his own variant: tie-break instead of discard-on-disagreement.
# It fires ONLY when the primary path cannot date the item, and ONLY for sources
# that declare a `dateline` block. A source with no such block behaves exactly as
# it did before the amendment.
#
# The standing caveat, restated where the code lives: a dateline is CONTENT, and
# content is what a sloppy or adversarial source controls. `Last-Modified` is at
# least an infrastructure fact. The mitigations are that both legs must agree or
# be tie-broken conservatively, that the URL leg is structural rather than
# textual, that a future-dated result is refused outright (DL-6), and that the
# conflict rate is counted. None of that makes such an item as good as one
# carrying a real release instant, and none of it should ever be described as if
# it did.
MODE_LIVE = "live"
MODE_RECONSTRUCTION = "reconstruction"

_MONTH_NAMES = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}
_MONTH_ABBR = {m[:3].lower(): i for m, i in _MONTH_NAMES.items()}


def _month_num(raw: str) -> int | None:
    s = (raw or "").strip().lower().rstrip(".")
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return _MONTH_NAMES.get(s) or _MONTH_ABBR.get(s[:3])


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    return first, nxt - timedelta(days=1)


def parse_url_date(url: str, pattern: str) -> tuple[date, date] | None:
    """DL-2 leg 1 — declared URL regex -> (earliest, latest) day consistent with it.

    Month-precision URLs give the whole month; a day-precision URL collapses to
    that day. Returns None when the declared pattern does not match, which DL-5
    treats as a missing leg.
    """
    m = re.search(pattern, url or "")
    if not m:
        return None
    g = m.groupdict()
    try:
        year = int(g.get("year") or 0)
        month = _month_num(g.get("month") or g.get("month_name") or "")
        if not year or not month:
            return None
        if g.get("day"):
            d = date(year, month, int(g["day"]))
            return d, d
        return _month_bounds(year, month)
    except (ValueError, TypeError):
        return None


def parse_body_dateline(body: str, pattern: str) -> date | None:
    """DL-2 leg 2 — FIRST match of the declared per-source pattern -> a calendar day.

    First-match, not best-match: the declared pattern is anchored per source and
    the chrome-strip puts a document's own dateline near the top, so the first
    hit is the dateline rather than a date mentioned in the prose. Choosing among
    later matches would be an inference, and inference is deferred under DL-5.
    """
    m = re.search(pattern, body or "")
    if not m:
        return None
    g = m.groupdict()
    try:
        year = int(g.get("year") or 0)
        month = _month_num(g.get("month") or g.get("month_name") or "")
        day = int(g.get("day") or 0)
        if not (year and month and day):
            return None
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def _eod(d: date) -> datetime:
    """The conservative convention: a day-precision date is 23:59:59Z of that day."""
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def resolve_dateline(spec: dict, url: str, body: str,
                     mode: str = MODE_LIVE) -> tuple[datetime | None, bool, str]:
    """DL-3/DL-4/DL-5 -> (instant, conflict, reason-it-failed).

    DL-3 agree      -> the body day at 23:59:59Z.
    DL-4 disagree   -> tie-break, never discard:
                       live           = the EARLIEST day either leg allows
                                        (a stale document must not read as fresh
                                        — the centcom.mil failure mode),
                       reconstruction = the LATEST day either leg allows
                                        (nothing enters a packet dated before it
                                        existed — the GDELT/release_timestamp rule).
    DL-5 either leg missing -> discard, with the reason recorded.
    """
    u = parse_url_date(url, spec.get("url_pattern", ""))
    b = parse_body_dateline(body, spec.get("body_pattern", ""))
    if u is None and b is None:
        return None, False, ("neither a URL-path date nor a body dateline could be "
                             "parsed — single-source dating is not permitted (DL-5)")
    if u is None:
        return None, False, ("no URL-path date on this document — the structural leg "
                             "is missing and a body dateline alone cannot date an "
                             "item (DL-5; context-based inference is deferred)")
    if b is None:
        return None, False, ("no body dateline matched the declared pattern — the "
                             "content leg is missing (DL-5)")
    lo, hi = u
    if lo <= b <= hi:
        return _eod(b), False, ""
    # DL-4: they disagree. Keep the item; take the bound that cannot flatter us.
    if mode == MODE_RECONSTRUCTION:
        return _eod(max(b, hi)), True, ""
    return _eod(min(b, lo)), True, ""


def prose_paragraphs(body: str) -> list[str]:
    """Body -> its prose lines (headings and datelines excluded).

    Used for `trail`/`lede`/`last_para`, which are the text Stage A actually
    reads. Splitting on blank lines is wrong here because the chrome-strip already
    removed them; a document's paragraphs are its long lines.
    """
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    prose = [ln for ln in lines
             if not _HEADING_RE.match(ln)
             and len(re.sub(r"[*_`#>]", "", ln).split()) >= PROSE_WORDS_MIN]
    return prose or lines or [body]


def item_from_jina(job: dict, raw: str, cutoff: datetime,
                   mode: str = MODE_LIVE,
                   fetch_instant: datetime | None = None) -> tuple[dict | None, str]:
    """Raw jina response + its worklist job -> an article-shaped item, or absence.

    `job` comes from the WORKLIST, not from the agent, so `source_type`, `domain`
    and `display` are decided by curated config and cannot be smuggled in with the
    payload.
    """
    if job.get("role") == "index":
        return None, ("rolling/index page — its Last-Modified is a touch time, not "
                      "an event time; index pages are navigation only, never items")
    if job.get("role") != "document":
        return None, f"job role {job.get('role')!r} is not a jina document job"
    parsed, why = parse_jina(raw)
    if parsed is None:
        return None, why
    body = parsed["body"]
    trusted = bool(job.get("timestamp_trusted"))
    dl_spec = job.get("dateline")
    dt = parsed["published"] if trusted else None
    conflict = False
    dated_by = "published_time"

    if dt is None:
        # DL-1: the primary path could not date this item — either the header is
        # absent, or the source's measured class says the header is not a
        # publication instant. The second path fires only if this source declares
        # one; otherwise the pre-amendment refusal stands, verbatim.
        if not dl_spec:
            if not trusted:
                return None, (
                    f"the Published Time header on {job.get('domain', 'this source')} "
                    f"is not a verified publication instant (measured class: "
                    f"{job.get('timestamp_class', 'unmeasured')!r}) — a header that "
                    "tracks the fetch clock or is absent cannot date evidence")
            return None, ("no usable Published Time header — undated pages are not "
                          "evidence (the nobelprize.org case)")
        dt, conflict, why = resolve_dateline(dl_spec, job.get("url", ""), body, mode)
        if dt is None:
            return None, why
        dated_by = "dateline_conflict" if conflict else "dateline"
        # DL-6, non-negotiable: a CONTENT-derived date is the one kind that can
        # invent a document from the future. Refused outright, with its own reason.
        horizon = min(x for x in (fetch_instant, cutoff) if x is not None)
        if dt > horizon:
            return None, (f"dateline resolves to {_compact(dt)}, later than the fetch "
                          f"instant {_compact(horizon)} — a future-dated document is "
                          "refused outright (DL-6)")

    if dt > cutoff:
        return None, f"published {_compact(dt)} is after the run cutoff {_compact(cutoff)}"
    paras = prose_paragraphs(body)
    title = content_unique_title(parsed["title"], dt, body,
                                 bool(job.get("immutable", False)))
    return ({"title": title[:300],
             "seendate": _compact(dt),
             "domain": job["domain"],
             "url": job.get("url") or parsed["url"],
             "trail": paras[0][:300],
             "lede": paras[0][:800],
             "last_para": paras[-1][:500] if len(paras) > 1 else "",
             "scan_text": body,              # internal keyword-match surface only
             "reach": True,
             "reach_kind": "document",
             "reach_source_type": job["source_type"],
             "reach_slugs": list(job.get("slugs", [])),
             "dated_by": dated_by,
             "dateline_conflict": conflict,
             # Official primary documents display headline + source + link with the
             # body internal-only — the pdf_ingest treatment, APPROVED 2026-08-24.
             "display": True}, "")


def item_from_transcript(job: dict, info: dict, vtt: str,
                         cutoff: datetime) -> tuple[dict | None, str]:
    """yt-dlp `--dump-json` + the .vtt -> an article-shaped item, or absence."""
    if job.get("role") != "video":
        return None, f"job role {job.get('role')!r} is not a transcript job"
    live = (info or {}).get("live_status")
    if live in ("is_live", "post_live", "is_upcoming"):
        return None, f"live_status={live} — content is still accruing"
    upload = info.get("timestamp")
    release = info.get("release_timestamp")
    duration = info.get("duration") or 0
    candidates = [t for t in (upload, (release + duration) if release else None)
                  if isinstance(t, (int, float)) and t > 0]
    if not candidates:
        return None, ("no upload or release timestamp — a transcript is NEVER dated "
                      "by the event date in its title")
    dt = datetime.fromtimestamp(max(candidates), tz=timezone.utc)
    if dt > cutoff:
        return None, f"transcript instant {_compact(dt)} is after the cutoff {_compact(cutoff)}"
    text = strip_vtt(vtt)
    words = len(text.split())
    if words < MIN_TRANSCRIPT_WORDS:
        return None, (f"only {words} words after cue-strip (floor "
                      f"{MIN_TRANSCRIPT_WORDS}) — an empty caption response is a "
                      "known transient, not proof the video has no captions")
    raw_title = (info.get("title") or job.get("label") or "transcript").strip()
    title = content_unique_title(raw_title, dt, text, immutable=False)
    return ({"title": title[:300],
             "seendate": _compact(dt),
             "domain": job["domain"],
             "url": info.get("webpage_url") or job.get("url", ""),
             "trail": text[:300],
             "lede": text[:800],
             "last_para": text[-500:] if len(text) > 1300 else "",
             "scan_text": text[:SCAN_CHARS],
             "reach": True,
             "reach_kind": "transcript",
             "reach_source_type": job["source_type"],
             "reach_slugs": list(job.get("slugs", [])),
             "reach_words": words,
             # A caption track is a full copy of the work. Always internal, on
             # every source, regardless of who published it.
             "display": False}, "")


# --------------------------------------------------------------------------
# The worklist
# --------------------------------------------------------------------------
def load_sources(path: Path | None = None) -> dict:
    p = path or SOURCES_FILE
    return json.loads(p.read_text())


def _resolve_template(tpl: str, d: datetime) -> str:
    return (tpl.replace("{YYYYMMDD}", d.strftime("%Y%m%d"))
               .replace("{YYYY-MM-DD}", d.strftime("%Y-%m-%d"))
               .replace("{YYYY}", d.strftime("%Y"))
               .replace("{MM}", d.strftime("%m"))
               .replace("{DD}", d.strftime("%d")))


def trigger_fires(trig: dict, date: str, calendars: dict) -> tuple[bool, datetime | None]:
    """Does this source fire on `date`? Returns (fires, the anchor date).

    The anchor is what a dated URL template resolves against: for a calendar
    trigger it is the CALENDAR date, not the run date, so a T+1 fetch of an FOMC
    statement still builds the meeting-day URL.
    """
    run = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    kind = (trig or {}).get("kind", "always")
    if kind == "always":
        return True, run
    if kind == "weekday":
        return run.weekday() in set(trig.get("weekdays", [])), run
    if kind == "window":
        start, end = trig.get("start", "0000-01-01"), trig.get("end", "9999-12-31")
        return start <= date <= end, run
    if kind == "calendar":
        days = calendars.get(trig.get("calendar", ""), [])
        for offset in trig.get("offsets_days", [0]):
            for day in days:
                anchor = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                if (anchor + timedelta(days=offset)).strftime("%Y-%m-%d") == date:
                    return True, anchor
        return False, None
    return False, None


def job_id(slug: str, target: str, date: str) -> str:
    return hashlib.sha1(f"{slug}|{target}|{date}".encode()).hexdigest()[:12]


def _ytdlp_command(url: str) -> str:
    """The yt-dlp invocation that actually writes a VTT on this machine.

    Two gotchas, both hit live on 2026-08-24 and both silent:
      * `--dump-json` implies simulate, so it prints metadata and writes NO
        subtitle file. `--print-json` prints the same metadata AND writes files.
      * `--sub-lang en` (singular) matched nothing and exited 1 without a file;
        `--sub-langs "en.*"` is what selects the English track.
    """
    return (f'yt-dlp --skip-download --write-subs --write-auto-subs '
            f'--sub-langs "en.*" --sub-format vtt --print-json -o "%(id)s" \'{url}\'')


def build_worklist(date: str, live_slugs: list[str], doc: dict,
                   sources: dict | None = None) -> dict:
    """Today's reach jobs for the live slate, with the exact command per job.

    Three gates run here, in order, and each one records WHY it skipped so the run
    log never has to guess:

      1. `enabled: false` — the source was probed and measured unusable (no
         timestamp, Cloudflare-blocked, JS-only). Skipping costs nothing and
         avoids burning a fetch on a rejection we already know the answer to.
      2. the trigger — a source only fires on days it can carry information
         (FOMC day, a weekday briefing, an announcement window).
      3. the doctor channel gate — `web` for pages, `youtube` for transcripts.

    Jobs are deduplicated by resolved target, so one UKMTO fetch serves both
    Hormuz markets and one Conseil-constitutionnel fetch serves both French ones.
    """
    src = sources or load_sources()
    calendars = src.get("calendars", {})
    web_ok, web_why = channel_status(doc, "web")
    vid_ok, vid_why = channel_status(doc, "youtube")
    jobs: dict[str, dict] = {}
    skipped: list[dict] = []
    for slug in live_slugs:
        for entry in src.get("markets", {}).get(slug, []):
            role = entry.get("role")
            if role not in ROLES:
                raise ValueError(f"reach_sources: bad role {role!r} for {slug}")
            if entry.get("source_type") not in SOURCE_TYPES:
                raise ValueError(f"reach_sources: bad source_type "
                                 f"{entry.get('source_type')!r} for {slug}")
            if not entry.get("enabled", True):
                probe = entry.get("probe", {})
                skipped.append({"slug": slug, "label": entry["label"],
                                "channel": "config",
                                "why": f"disabled in reach_sources.json "
                                       f"(probe {probe.get('date', '?')}: "
                                       f"{probe.get('published_time', 'unusable')})"})
                continue
            fires, anchor = trigger_fires(entry.get("trigger", {"kind": "always"}),
                                          date, calendars)
            if not fires:
                continue
            anchor = anchor or datetime.fromisoformat(date)
            channel_ok, channel_why = (vid_ok, vid_why) if role == "video" else (web_ok, web_why)
            if not channel_ok:
                skipped.append({"slug": slug, "label": entry["label"],
                                "channel": "youtube" if role == "video" else "web",
                                "why": channel_why})
                continue
            if role == "video":
                target = entry.get("video_id") or entry.get("channel", "")
            else:
                target = _resolve_template(entry["url"], anchor)
            jid = job_id("shared", target, date)
            job = jobs.get(jid)
            if job is None:
                job = {"id": jid, "role": role, "label": entry["label"],
                       "domain": entry["domain"], "source_type": entry["source_type"],
                       "immutable": bool(entry.get("immutable", False)),
                       "why": entry.get("why", ""), "slugs": [],
                       "anchor_date": anchor.strftime("%Y-%m-%d")}
                if role == "video":
                    if entry.get("video_id"):
                        job["url"] = f"https://www.youtube.com/watch?v={entry['video_id']}"
                        job["command"] = _ytdlp_command(job["url"])
                    else:
                        # discovery job: the agent picks the matching upload off the
                        # channel and submits it as a child, which is provenance-
                        # checked against `channel` on ingest.
                        job["channel"] = entry["channel"]
                        job["match"] = entry.get("match", "")
                        job["url"] = entry.get("search", "")
                        job["command"] = (
                            f"yt-dlp --flat-playlist --playlist-end 10 "
                            f"--print '%(id)s | %(title)s' '{job['url']}'"
                            + (f"   # then fetch the upload matching "
                               f"{entry.get('match')!r}:" if entry.get("match") else "")
                            + f"\n    {_ytdlp_command('https://www.youtube.com/watch?v=<ID>')}")
                        job["child"] = {"kind": "video", "channel": entry["channel"],
                                        "enabled": True}
                else:
                    job["url"] = target
                    job["command"] = f"curl -s --max-time 60 '{JINA_PREFIX}{target}'"
                    ts_class = (entry.get("probe") or {}).get("published_time", "unmeasured")
                    job["timestamp_class"] = ts_class
                    job["timestamp_trusted"] = ts_class == TRUSTED_TIMESTAMP
                    if entry.get("dateline"):
                        job["dateline"] = dict(entry["dateline"])
                    if role == "index" and entry.get("child"):
                        job["child"] = dict(entry["child"])
                probe = entry.get("probe")
                if probe:
                    job["probe"] = probe
                jobs[jid] = job
            if slug not in job["slugs"]:
                job["slugs"].append(slug)
    ordered = sorted(jobs.values(), key=lambda j: (j["role"], j["label"]))
    return {"date": date, "generated_at": datetime.now(timezone.utc).isoformat(),
            "doctor_available": bool(doc.get("available")),
            "n_jobs": len(ordered), "jobs": ordered, "skipped": skipped,
            "how_to_reply": {
                "file": "reach_items.json (a JSON list) in this day dir",
                "direct_page": {"id": "<job id>", "kind": "web",
                                "raw": "<the full r.jina.ai response, or raw_file>"},
                "direct_video": {"id": "<job id>", "kind": "video",
                                 "info": "<yt-dlp --print-json output, or info_file>",
                                 "vtt": "<the .en.vtt text, or vtt_file>"},
                "discovered_page": {"parent": "<index job id>", "kind": "web",
                                    "url": "<the dated document URL found under it>",
                                    "raw": "<its r.jina.ai response, or raw_file>"},
                "discovered_video": {"parent": "<video job id>", "kind": "video",
                                     "info": "...", "vtt": "..."}},
            "rules": [
                "Raw payloads only. The validation gate is newsagent.reach, not you:"
                " do not pre-filter, do not fix up, do not summarise.",
                "`index` jobs are NAVIGATION. Read them to find today's dated"
                " document URL and submit that as a `parent`-linked child; never"
                " submit the index page itself as an item.",
                "A child page URL must start with its parent's declared url_prefix,"
                " and a child video must come from its parent's declared channel."
                " Anything else is rejected — curation stays in config.",
                "If a fetch fails or looks wrong, submit NOTHING for it. A failed"
                " item is absent; it is never a placeholder or an error string.",
            ]}


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def _read_payload(rec: dict, key: str, day_dir: Path) -> str:
    if rec.get(key) is not None:
        return rec[key] if isinstance(rec[key], str) else json.dumps(rec[key])
    fname = rec.get(f"{key}_file")
    if not fname:
        return ""
    p = Path(fname)
    if not p.is_absolute():
        p = day_dir / fname
    return p.read_text() if p.exists() else ""


def _child_job(parent: dict, rec: dict, info: dict | None) -> tuple[dict | None, str]:
    """Turn a `parent`-linked submission into a synthetic job, or refuse it.

    This is the provenance boundary. `domain`, `source_type` and `immutable` are
    ALWAYS inherited from the curated parent — the agent supplies bytes and, for a
    page, a URL that must sit under the parent's declared `url_prefix`. That is
    what keeps curation in config: a discovered document cannot arrive from a
    domain nobody curated, and cannot promote itself to a heavier source type.
    """
    child = parent.get("child")
    if not child:
        return None, (f"{parent['label']!r} declares no ingestable document shape — "
                      "it is navigation only")
    if not child.get("enabled", True):
        return None, (f"documents under {parent['label']!r} are disabled: "
                      f"{child.get('why_disabled', 'measured unusable')}")
    base = {"domain": parent["domain"], "source_type": parent["source_type"],
            "slugs": parent.get("slugs", []), "label": parent["label"]}
    if child.get("kind") == "video":
        want = (child.get("channel") or "").lstrip("@").lower()
        got = str((info or {}).get("uploader_id") or "").lstrip("@").lower()
        alt = str((info or {}).get("channel_id") or "").lower()
        if want and want not in (got, alt):
            return None, (f"video is from {got or alt or 'an unknown channel'!r}, "
                          f"not the curated channel @{want}")
        return {**base, "role": "video",
                "label": (info or {}).get("title") or parent["label"],
                "url": (info or {}).get("webpage_url", "")}, ""
    url = (rec.get("url") or "").strip()
    prefix = child.get("url_prefix", "")
    if not url:
        return None, "a discovered document must carry the url it was fetched from"
    if prefix and not url.startswith(prefix):
        return None, (f"url is outside the curated prefix for {parent['label']!r} "
                      f"({prefix}) — curation stays in config, not in the payload")
    ts_class = child.get("published_time", parent.get("timestamp_class", "unmeasured"))
    job = {**base, "role": "document", "url": url,
           "immutable": bool(child.get("immutable", False)),
           "timestamp_class": ts_class,
           "timestamp_trusted": ts_class == TRUSTED_TIMESTAMP}
    # A discovered document inherits its parent's declared dateline spec — the
    # curation stays in config, exactly as domain/source_type/immutable do.
    dl = child.get("dateline") or parent.get("dateline")
    if dl:
        job["dateline"] = dl
    return job, ""


def ingest_reach_file(path: Path, worklist: dict, day_dir: Path,
                      cutoff: datetime, mode: str = MODE_LIVE
                      ) -> tuple[list[dict], list[dict]]:
    """Raw agent payloads -> (validated items, rejections).

    Every rejection is recorded with its reason and NOTHING is written for it —
    the item is absent from the packet, not degraded into it. The Reuters probe is
    the cautionary case the whole gate exists for: a naive fetch would have written
    `AbuseAlleviationError: … DDoS attack suspected` into the packet as evidence
    text, and Stage A would have dutifully extracted features from it.
    """
    records = json.loads(Path(path).read_text())
    if isinstance(records, dict):
        records = records.get("items", [])
    by_id = {j["id"]: j for j in worklist.get("jobs", [])}
    items, rejected = [], []
    for rec in records:
        parent_id, own_id = rec.get("parent"), rec.get("id")
        parent = by_id.get(parent_id) if parent_id else None
        job = by_id.get(own_id) if own_id else None
        if parent is None and job is None:
            rejected.append({"id": own_id or parent_id, "label": rec.get("label", "?"),
                             "why": "no matching job in today's worklist — reach items "
                                    "must come from the curated worklist"})
            continue

        info: dict | None = None
        wants_video = (parent or job).get("role") == "video" or \
                      ((parent or {}).get("child") or {}).get("kind") == "video"
        if wants_video:
            raw_info = _read_payload(rec, "info", day_dir)
            try:
                info = json.loads(raw_info) if raw_info else {}
            except Exception:
                info = {}

        if parent is not None:
            job, why = _child_job(parent, rec, info)
            if job is None:
                rejected.append({"id": parent_id, "label": parent["label"], "why": why})
                continue

        if job["role"] == "video":
            if not info:
                item, why = None, "no yt-dlp --print-json metadata supplied"
            else:
                item, why = item_from_transcript(
                    job, info, _read_payload(rec, "vtt", day_dir), cutoff)
        else:
            fetched = parse_published(rec.get("fetched_at", "")) if rec.get("fetched_at") else None
            item, why = item_from_jina(job, _read_payload(rec, "raw", day_dir),
                                       cutoff, mode=mode, fetch_instant=fetched)

        if item is None:
            rejected.append({"id": own_id or parent_id, "label": job.get("label", "?"),
                             "why": why})
        else:
            item["reach_job"] = own_id or parent_id
            items.append(item)
    return items, rejected


def window_start(kind: str, now: datetime) -> datetime:
    """Declared per-kind window: 14d for official documents, 72h for transcripts."""
    return now - (timedelta(days=OFFICIAL_DOC_WINDOW_DAYS) if kind == "document"
                  else timedelta(hours=TRANSCRIPT_WINDOW_HOURS))


def items_for_slug(items: list[dict], slug: str) -> list[dict]:
    """Reach items are CURATED PER MARKET: an item only reaches the packet of a
    market that asked for it, even if its text happens to match another market's
    keywords. Keyword + window filtering still applies on top."""
    return [a for a in items if slug in (a.get("reach_slugs") or [])]


def reach_items_path(date: str) -> Path:
    return DATA / date / "reach_items.json"
