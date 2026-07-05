"""Public macro-research PDF ingestion (v3.1 source — lifted from the n8n macro pipeline).

Five publicly-accessible weekly/periodic research PDFs, fetched by URL with no
login (we pay for none of them): JPM Weekly Market Recap, JPM Guide to the
Markets (UK), JPM Weekly Brief (static URLs), GS Weekly Market Monitor
(date-templated: most recent Friday, MMDDYY) and BofA Capital Markets Outlook
(most recent Monday, MM-DD-YYYY).

Rules (build notes + Justin's display line):
  - Full text is used INTERNALLY for Stage-A extraction only. On the public page
    a PDF item appears in the evidence feed as headline + source + link
    (aggregator style) — never a verbatim re-host of the body.
  - A per-source DISCLAIMER-STRIP runs before any text reaches Stage A
    (boilerplate/legal lines would otherwise pollute feature extraction).
  - Items are macro-tagged ("macro": True) — strong for macro questions
    (Fed/rates), lighter for pure politics; the per-market keyword filter decides.
  - DEGRADE SILENTLY: a missing weekly PDF (not yet published / URL rotated)
    skips that source with a one-line note; the pipeline never breaks. GS/BofA
    also retry one week back before giving up.
  - Live-only source: like RSS/newsletters there is no timestamped archive at
    these URLs, so PDFs are excluded from historical packet reconstruction by
    construction.

Source weights: these research desks are OUT OF SCOPE of the RSP tiers and run
at the neutral 1.0 until Justin signs off a tier mapping (same rule as ING
newsletters — see the uncovered-source proposal table in the findings note).
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import DATA

CACHE_DIR = DATA / "pdf_cache"
MAX_PAGES = 6          # text pages read per PDF (decks are chart-heavy past that)
MAX_CHARS = 4000       # internal scan text cap per report

PDF_SOURCES = [
    {"label": "JPM Weekly Market Recap", "domain": "am.jpmorgan.com", "dated": None,
     "url": ("https://am.jpmorgan.com/content/dam/jpm-am-aem/americas/us/en/insights/"
             "market-insights/wmr/weekly_market_recap.pdf")},
    {"label": "JPM Guide to the Markets (UK)", "domain": "jpmorganfunds.com", "dated": None,
     "url": ("https://cdn.jpmorganfunds.com/content/dam/jpm-am-aem/global/en/insights/"
             "market-insights/guide-to-the-markets/mi-guide-to-the-markets-uk.pdf")},
    {"label": "JPM Weekly Brief", "domain": "am.jpmorgan.com", "dated": None,
     "url": ("https://am.jpmorgan.com/content/dam/jpm-am-aem/emea/regional/en/insights/"
             "market-insights/the-weekly-brief/mi-weekly-market-brief-en.pdf")},
    {"label": "GS Weekly Market Monitor", "domain": "am.gs.com", "dated": "friday",
     "url": ("https://am.gs.com/cms-assets/gsam-app/documents/insights/en/{YYYY}/"
             "market_monitor_{MMDDYY}.pdf?view=true")},
    {"label": "BofA Capital Markets Outlook", "domain": "ml.com", "dated": "monday",
     "url": ("https://ustrustaem.fs.ml.com/content/dam/ust/ecomm/pdf/"
             "CMO_Institutional_{MM-DD-YYYY}_ada.pdf")},
]

# Disclaimer/boilerplate strip (pattern reused from the n8n per-source strip step):
# any LINE matching one of these is dropped before Stage A sees the text.
DISCLAIMER_PATTERNS = [
    r"(?i)past performance", r"(?i)not (a|an) (solicitation|offer|recommendation)",
    r"(?i)for institutional", r"(?i)investment professionals? only",
    r"(?i)all rights reserved", r"©|\(c\) \d{4}", r"(?i)important (disclos|information)",
    r"(?i)member (finra|sipc)", r"(?i)fdic", r"(?i)may lose value",
    r"(?i)not bank guaranteed", r"(?i)opinions .{0,40}subject to change",
    r"(?i)should not be (relied|construed)", r"(?i)distribut(ed|ion) .{0,30}prohibited",
    r"(?i)risk of loss", r"(?i)consult .{0,30}(advisor|professional)",
    r"(?i)^\s*(disclosure|disclaimer)s?\b",
]
_DISC_RE = [re.compile(p) for p in DISCLAIMER_PATTERNS]


def most_recent(weekday: int, t: datetime) -> datetime:
    """Most recent date with datetime.weekday()==weekday, at or before t."""
    return t - timedelta(days=(t.weekday() - weekday) % 7)


def source_urls(src: dict, t: datetime) -> list[str]:
    """Concrete URL candidates for a source at time t (dated ones retry -1 week)."""
    if not src["dated"]:
        return [src["url"]]
    anchor = most_recent(4 if src["dated"] == "friday" else 0, t)
    out = []
    for back in (0, 7):
        d = anchor - timedelta(days=back)
        out.append(src["url"].replace("{YYYY}", d.strftime("%Y"))
                             .replace("{MMDDYY}", d.strftime("%m%d%y"))
                             .replace("{MM-DD-YYYY}", d.strftime("%m-%d-%Y")))
    return out


def strip_disclaimers(text: str) -> str:
    """Drop boilerplate/legal lines before Stage-A extraction."""
    kept = [ln for ln in text.splitlines()
            if ln.strip() and not any(rx.search(ln) for rx in _DISC_RE)]
    out = "\n".join(kept)
    out = re.sub(r"[ \t]+", " ", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for page in reader.pages[:MAX_PAGES]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks)


def _fetch_one(src: dict, t: datetime) -> dict | None:
    """One source -> one article-shaped item, or None (degrade silently)."""
    for url in source_urls(src, t):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            data = urllib.request.urlopen(req, timeout=60).read()
            text = strip_disclaimers(_pdf_text(data))[:MAX_CHARS]
        except Exception:
            continue
        if len(text) < 200:   # scanned/chart-only or extraction failure — not usable
            continue
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 60] or [text]
        return {"title": f"{src['label']} — week of {t.strftime('%Y-%m-%d')}",
                "seendate": t.strftime("%Y%m%dT120000Z"),
                "domain": src["domain"], "url": url.split("?")[0],
                "trail": paras[0][:300], "lede": paras[0][:800],
                "last_para": paras[-1][:500] if len(paras) > 1 else "",
                "scan_text": text,      # internal keyword-match surface only
                "macro": True}
    return None


def fetch_pdf_reports(day: str | None = None) -> list[dict]:
    """All macro-research PDF items for the run day, fetched once (day-cached).

    Missing/unparseable reports degrade to absence — never an exception."""
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{day}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        import pypdf  # noqa: F401
    except ImportError:
        print("  pdf: pypdf not installed — macro PDFs skipped (uv add pypdf)")
        return []
    t = datetime.fromisoformat(day).replace(hour=12, tzinfo=timezone.utc)
    items, missing = [], []
    for src in PDF_SOURCES:
        item = _fetch_one(src, t)
        if item is not None:
            items.append(item)
        else:
            missing.append(src["label"])
    if missing:
        print(f"  pdf: {len(missing)} report(s) unavailable this week "
              f"({', '.join(missing)}) — degraded silently")
    cache.write_text(json.dumps(items))
    return items
