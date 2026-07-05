"""Read-only newsletter ingestion from the epsilonfund inbox (v3 source).

Newsletters (ING daily + weekly macro to start) are ANALYSIS-GRADE input for
Stage A — richer than headlines. They are a *source*, never the scheduler, and
their content is private/licensed: items carry display=False and the public page
shows only a generic source label (never title, text, or link). IP-scrub rule.

Credential (Justin provides; secrets/ is gitignored — never commit):
  EITHER secrets/email_imap.json      {"host", "user", "app_password", "mailbox"?}
  OR     secrets/gmail_oauth.json     (Gmail API OAuth client, readonly scope) +
         secrets/gmail_token.json     (user token minted once, attended)
IMAP is the simpler path (app password on the workspace account); the Gmail API
path needs `google-api-python-client` + an attended OAuth dance — scaffolded via
the same available()/fetch interface, IMAP implemented first.

Everything is read-only: IMAP opens the mailbox with readonly=True; no message is
ever marked, moved, or deleted. Absent credential -> available()=(False, why) and
the pipeline runs without newsletters (additive by construction).
"""
from __future__ import annotations

import email
import email.policy
import imaplib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DATA, ROOT

IMAP_CRED = ROOT.parents[1] / "secrets" / "email_imap.json"
GMAIL_CRED = ROOT.parents[1] / "secrets" / "gmail_oauth.json"
CACHE_DIR = DATA / "newsletter_cache"

# Sender filters (extend as more newsletters are approved). Matching is on the
# From header, case-insensitive substring.
NEWSLETTER_FILTERS = [
    {"sender_contains": "ing.com", "label": "ING research"},
    {"sender_contains": "think.ing.com", "label": "ING THINK"},
]


def available() -> tuple[bool, str]:
    if IMAP_CRED.exists():
        return True, "imap"
    if GMAIL_CRED.exists():
        return False, ("gmail OAuth client found but the Gmail API path is scaffold-only "
                       "— use IMAP (secrets/email_imap.json) for now")
    return False, ("no email credential — Justin: create a read-only app password for "
                   "the epsilonfund inbox and save secrets/email_imap.json "
                   '{"host": "imap.gmail.com", "user": "...", "app_password": "..."}')


def _html_to_text(html_body: str) -> str:
    txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_body, flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", txt, flags=re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"&nbsp;", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return re.sub(r"\n{2,}", "\n\n", txt).strip()


def _body_text(msg: email.message.EmailMessage) -> str:
    plain = msg.get_body(preferencelist=("plain",))
    if plain is not None:
        return plain.get_content().strip()
    html_part = msg.get_body(preferencelist=("html",))
    if html_part is not None:
        return _html_to_text(html_part.get_content())
    return ""


def parse_message(raw: bytes) -> dict | None:
    """One email -> one article-shaped newsletter item (display=False)."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    sender = str(msg.get("From", ""))
    if not any(f["sender_contains"].lower() in sender.lower() for f in NEWSLETTER_FILTERS):
        return None
    label = next(f["label"] for f in NEWSLETTER_FILTERS
                 if f["sender_contains"].lower() in sender.lower())
    subject = str(msg.get("Subject", "")).strip()
    try:
        dt = email.utils.parsedate_to_datetime(str(msg.get("Date", "")))
        seendate = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        seendate = ""
    body = _body_text(msg)
    if not subject or not body:
        return None
    paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 80]
    lede = paras[0][:800] if paras else body[:800]
    last = paras[-1][:500] if len(paras) > 1 else ""
    return {"title": subject[:200], "seendate": seendate,
            "domain": f"newsletter:{label}", "url": "",
            "trail": "", "lede": lede, "last_para": last,
            "display": False}   # NEVER shown publicly — Stage-A input only


def fetch_newsletters(since_days: int = 3, day: str | None = None) -> list[dict]:
    """Newsletter items from the inbox via read-only IMAP; day-cached.

    Returns [] when no credential is present (pipeline stays additive)."""
    ok, mode = available()
    if not ok:
        return []
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{day}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    cred = json.loads(IMAP_CRED.read_text())
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%d-%b-%Y")
    items = []
    conn = imaplib.IMAP4_SSL(cred.get("host", "imap.gmail.com"))
    try:
        conn.login(cred["user"], cred["app_password"])
        conn.select(cred.get("mailbox", "INBOX"), readonly=True)   # read-only, always
        for f in NEWSLETTER_FILTERS:
            typ, data = conn.search(None, f'(SINCE "{since}" FROM "{f["sender_contains"]}")')
            if typ != "OK" or not data or not data[0]:
                continue
            for num in data[0].split()[-10:]:   # cap per filter
                typ, msg_data = conn.fetch(num, "(BODY.PEEK[])")   # PEEK: no flag changes
                if typ != "OK" or not msg_data or msg_data[0] is None:
                    continue
                item = parse_message(msg_data[0][1])
                if item:
                    items.append(item)
    finally:
        try:
            conn.logout()
        except Exception:
            pass
    # dedupe by (title, date-day)
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x.get("seendate", ""), reverse=True):
        key = (it["title"].lower(), it.get("seendate", "")[:8])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    cache.write_text(json.dumps(out))
    return out
