"""Read-only newsletter ingestion from the epsilonfund inbox (v3 source; v3.1 Gmail API).

Newsletters (ING daily + weekly macro, Bloomberg) are ANALYSIS-GRADE input for
Stage A — richer than headlines. They are a *source*, never the scheduler, and
their content is private/licensed: items carry display=False and the public page
shows only a generic source label (never title, text, or link). IP-scrub rule.

Credential paths (secrets/ is gitignored — never commit), preferred first:
  1. Gmail API OAuth (v3.1, the path Justin set up):
       secrets/gmail_client_secret.json   installed-app OAuth client (Justin created)
       secrets/gmail_token.json           minted ONCE by the attended consent flow:
         PYTHONPATH=. uv run python -m newsagent.email_ingest --consent
       Scope is https://www.googleapis.com/auth/gmail.readonly — the token can
       never mark, move, send, or delete mail. Stdlib-only (auth-code + refresh
       grants over urllib; Gmail REST for list/get) — no google-api-python-client.
  2. IMAP app-password (v3 default, kept as fallback):
       secrets/email_imap.json            {"host", "user", "app_password", "mailbox"?}

Everything is read-only: Gmail scope is readonly; IMAP opens the mailbox with
readonly=True and fetches via BODY.PEEK. Absent credential -> available()=(False,
why) and the pipeline runs without newsletters (additive by construction).
"""
from __future__ import annotations

import argparse
import base64
import email
import email.policy
import http.server
import imaplib
import json
import re
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DATA, ROOT

SECRETS = ROOT.parents[1] / "secrets"
IMAP_CRED = SECRETS / "email_imap.json"
GMAIL_CLIENT_SECRET = SECRETS / "gmail_client_secret.json"
GMAIL_TOKEN = SECRETS / "gmail_token.json"
CACHE_DIR = DATA / "newsletter_cache"

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me/"

# Sender filters (extend as more newsletters are approved). Matching is on the
# From header, case-insensitive substring; the Gmail path also uses each entry as
# a `from:<sender> newer_than:Nd` query. economics.ingthink.com is ING THINK's
# actual sending domain (the bare "ing.com" substring does NOT match it).
NEWSLETTER_FILTERS = [
    {"sender_contains": "economics.ingthink.com", "label": "ING THINK"},
    {"sender_contains": "think.ing.com", "label": "ING THINK"},
    {"sender_contains": "ing.com", "label": "ING research"},
    {"sender_contains": "news.bloomberg.com", "label": "Bloomberg"},
]


def available() -> tuple[bool, str]:
    if GMAIL_TOKEN.exists():
        return True, "gmail"
    if IMAP_CRED.exists():
        return True, "imap"
    if GMAIL_CLIENT_SECRET.exists():
        return False, ("gmail OAuth client present but no token yet — run the one-time "
                       "read-only consent (attended, needs a browser signed in as the "
                       "epsilonfund account): PYTHONPATH=. uv run python -m "
                       "newsagent.email_ingest --consent")
    return False, ("no email credential — Justin: either place the Gmail OAuth client at "
                   "secrets/gmail_client_secret.json and run the consent flow, or create "
                   "a read-only app password at secrets/email_imap.json "
                   '{"host": "imap.gmail.com", "user": "...", "app_password": "..."}')


# ------------------------------------------------------------- Gmail API path --

def _client_secret() -> dict:
    return json.loads(GMAIL_CLIENT_SECRET.read_text())["installed"]


def _token_request(data: dict) -> dict:
    cs = _client_secret()
    body = urllib.parse.urlencode({**data, "client_id": cs["client_id"],
                                   "client_secret": cs["client_secret"]}).encode()
    req = urllib.request.Request(cs["token_uri"], data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _save_token(tok: dict, prev: dict | None = None) -> dict:
    """Persist the token file; Google omits refresh_token on refresh — keep the old one."""
    rec = {"access_token": tok["access_token"],
           "refresh_token": tok.get("refresh_token") or (prev or {}).get("refresh_token", ""),
           "scope": tok.get("scope", GMAIL_SCOPE),
           "expires_at": time.time() + float(tok.get("expires_in", 3600)) - 60}
    GMAIL_TOKEN.write_text(json.dumps(rec, indent=1))
    GMAIL_TOKEN.chmod(0o600)
    return rec


def run_consent(port: int = 8765, open_browser: bool = True, timeout: int = 600) -> None:
    """One-time ATTENDED read-only consent: prints/opens the Google auth URL, catches
    the localhost redirect, exchanges the code, writes secrets/gmail_token.json.

    Justin must complete the browser step signed in as the epsilonfund account."""
    cs = _client_secret()
    redirect = f"http://localhost:{port}"
    url = cs["auth_uri"] + "?" + urllib.parse.urlencode({
        "client_id": cs["client_id"], "redirect_uri": redirect,
        "response_type": "code", "scope": GMAIL_SCOPE,
        "access_type": "offline", "prompt": "consent"})
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got["code"] = (q.get("code") or [""])[0]
            got["error"] = (q.get("error") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            msg = "Consent received — you can close this tab." if got["code"] \
                else f"Consent failed: {got['error']}"
            self.wfile.write(f"<h3>{msg}</h3>".encode())

        def log_message(self, *a):  # silence request logging
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    srv.timeout = timeout
    print("Gmail read-only consent — open this URL signed in as the epsilonfund "
          "account:\n\n" + url + "\n\nWaiting for the browser redirect "
          f"(localhost:{port}, up to {timeout}s) ...")
    if open_browser:
        webbrowser.open(url)
    srv.handle_request()
    srv.server_close()
    if not got.get("code"):
        raise RuntimeError(f"consent not granted ({got.get('error') or 'timeout'}) — rerun "
                           "python -m newsagent.email_ingest --consent")
    tok = _token_request({"grant_type": "authorization_code", "code": got["code"],
                          "redirect_uri": redirect})
    _save_token(tok)
    print(f"token written -> {GMAIL_TOKEN} (read-only scope; refresh handled automatically)")


def _gmail_access_token() -> str:
    tok = json.loads(GMAIL_TOKEN.read_text())
    if time.time() >= float(tok.get("expires_at", 0)):
        tok = _save_token(_token_request({"grant_type": "refresh_token",
                                          "refresh_token": tok["refresh_token"]}), prev=tok)
    return tok["access_token"]


def _gmail_api(path: str, token: str, params: dict | None = None) -> dict:
    url = GMAIL_API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _fetch_gmail(since_days: int) -> list[dict]:
    """Newsletter items via the Gmail REST API (readonly scope, raw RFC822 -> the
    same parse_message as IMAP)."""
    token = _gmail_access_token()
    items, seen_ids = [], set()
    for f in NEWSLETTER_FILTERS:
        q = f'from:{f["sender_contains"]} newer_than:{since_days}d'
        try:
            res = _gmail_api("messages", token, {"q": q, "maxResults": "10"})
        except Exception:
            continue
        for m in res.get("messages", []) or []:
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])
            try:
                raw = _gmail_api(f"messages/{m['id']}", token, {"format": "raw"})
                data = base64.urlsafe_b64decode(raw["raw"] + "===")
            except Exception:
                continue
            item = parse_message(data)
            if item:
                items.append(item)
    return items


# ------------------------------------------------------------------ parsing ----

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


def _fetch_imap(since_days: int) -> list[dict]:
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
    return items


def fetch_newsletters(since_days: int = 3, day: str | None = None) -> list[dict]:
    """Newsletter items from the inbox (Gmail API preferred, IMAP fallback); day-cached.

    Returns [] when no credential is present, and ALSO when a credential is present
    but no longer works — a revoked/expired OAuth refresh token, a changed app
    password, an inbox outage. This channel is additive: a dead credential must
    degrade the packet, never break the daily run (same contract as `_refresh_gdelt`
    and the macro-PDF fetch). The reason is printed so an attended run can see it.

    Failure observed 2026-08-24: Google returned HTTP 400 on the refresh-token
    grant after the token sat unused since 2026-07-05 (unverified/testing OAuth
    apps expire refresh tokens in ~7 days) — before this guard it raised straight
    out of `--stage fetch` and stopped the run.
    """
    ok, mode = available()
    if not ok:
        return []
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{day}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        items = _fetch_gmail(since_days) if mode == "gmail" else _fetch_imap(since_days)
    except Exception as e:
        print(f"  newsletters: skipped — {mode} credential failed ({type(e).__name__}: "
              f"{str(e)[:120]}). Re-run the attended consent: "
              "PYTHONPATH=. uv run python -m newsagent.email_ingest --consent")
        return []
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consent", action="store_true",
                    help="run the one-time attended Gmail read-only consent flow")
    ap.add_argument("--no-browser", action="store_true",
                    help="print the auth URL instead of opening a browser")
    ap.add_argument("--probe", action="store_true",
                    help="fetch newsletters now (no day-cache) and print counts per label")
    args = ap.parse_args()
    if args.consent:
        run_consent(open_browser=not args.no_browser)
    elif args.probe:
        ok, mode = available()
        print(f"available: {ok} ({mode})")
        if ok:
            got = _fetch_gmail(7) if mode == "gmail" else _fetch_imap(7)
            counts: dict[str, int] = {}
            for it in got:
                counts[it["domain"]] = counts.get(it["domain"], 0) + 1
            print(json.dumps(counts, indent=1) or "no items")
    else:
        print("nothing to do — pass --consent or --probe")
