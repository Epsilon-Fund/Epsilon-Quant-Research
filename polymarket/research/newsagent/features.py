"""Stage A — cheap-LLM structured feature extraction (language -> features).

One extraction per (market, article), CACHED forever under
data/newsagent/live/feature_cache/<key>.json where key = sha1(slug + "|" + title).
Features are market-relative (relevance/stance/phase are w.r.t. the question), so
the cache key includes the market slug; the same article shared by two markets is
extracted once per market. Daily runs therefore only pay for genuinely new articles.

Execution paths (mirrors engine.py):
  1. Anthropic API with a CHEAP model (claude-haiku-4-5) — one batched call per
     pending group of <=BATCH articles for the same market.
  2. Gemini 2.5 Flash (v3.1, PROVIDER FLAG — free tier keeps the round zero-cost):
     same prompt/validation, selected via NEWSAGENT_EXTRACT_PROVIDER=gemini or
     --provider gemini; needs GEMINI_API_KEY. CAVEAT: alpha was fit on
     Haiku-era extraction — spot-check Gemini features against the existing
     cache (scripts/newsagent_provider_spotcheck.py) before leaning on it.
  3. Out-of-band: `pending_extractions()` emits a JSON work list; an operator or
     Claude Code subagents produce {cache_key: features} and `ingest_features()`
     validates + writes the cache. No API key needed.

The LLM never sees the market mid, any price, or the outcome — only question,
resolution criteria, and article text. Stage B (fvmodel.py) consumes the cache and
never calls an LLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

from .config import DATA, ANTHROPIC_KEY_ENV

CACHE_DIR = DATA / "feature_cache"
EXTRACT_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_KEY_ENV = "GEMINI_API_KEY"
PROVIDER_ENV = "NEWSAGENT_EXTRACT_PROVIDER"   # anthropic | gemini | oob (unset = auto)
BATCH = 12  # articles per extraction call (batched for cost; schema stays per-article)
# Prompt semantics version — part of the cache key, so a semantics change invalidates
# old records instead of silently mixing definitions. v2: stance = probability-impact
# on the QUESTION (v1 judged whether the subject welcomed the development, which
# inverted e.g. "challenger rises, leader vows to stay" on leader-exit questions).
# v3: adds per-article CLARITY (how unambiguous the item's signal is on the
# question) — feeds the band width, not the FV direction.
PROMPT_VERSION = 3

STANCES = {"toward_yes", "toward_no", "neutral"}
PHASES = {"none", "speculative", "planned", "in_progress", "completed"}
EVENT_TYPES = {"diplomacy", "conflict", "election", "personnel", "economic",
               "legal", "procedural", "other"}

EXTRACTION_PROMPT = """You are an information-extraction system. For a prediction-market question, extract structured features from each news item below. Judge each item ONLY on its own text. Do not forecast; do not guess an overall probability; extract per-item features only.

QUESTION: {question}
RESOLUTION CRITERIA (summary): {criteria}

For EACH item output one JSON object with fields:
- "id": the item id, copied verbatim.
- "relevance": 0.0-1.0 — how directly this item bears on the question (0 = unrelated).
- "stance": "toward_yes" | "toward_no" | "neutral" — probability impact: would a rational forecaster, reading ONLY this item, move P(question resolves YES) up (toward_yes), down (toward_no), or not at all (neutral)? Judge the impact on the QUESTION, not whether the subject of the news welcomes or resists the development. Example: for a "Leader X out by <date>" question, a rising challenger, growing party pressure, or a resignation demand is toward_yes even if X vows to stay; a consolidation of X's position is toward_no. If the item reports the qualifying event already happened (deal signed, person resigned, decision announced), that is toward_yes with event_phase "completed" and high strength.
- "event_phase": "none" | "speculative" | "planned" | "in_progress" | "completed" — the most advanced phase of the development this item credibly asserts, in EITHER direction. Rumor/opinion = speculative; scheduled/announced = planned; underway = in_progress; already happened = completed; no development bearing on the question = none.
- "strength": 0.0-1.0 — how decisive/credible the assertion is (specific, sourced, confirmed = high; vague speculation = low). 0 if event_phase is "none".
- "tone": -1.0..1.0 — overall tone of the item (negative..positive), independent of the question.
- "event_type": one of "diplomacy","conflict","election","personnel","economic","legal","procedural","other".
- "entities": up to 5 key named entities (people/orgs/places).
- "novelty": 0.0-1.0 — new development (1) vs rehash/background (0).
- "clarity": 0.0-1.0 — how UNAMBIGUOUS the item's signal is on the question: 1.0 = the implication for the question is unmistakable (single reading); 0.3 = mixed/contested signals or heavy hedging; 0.0 = the item's bearing on the question is anyone's guess. Independent of stance direction and of strength (a weak signal can still be perfectly clear).

ITEMS:
{items}

Reply with a single JSON array of one object per item, in the same order, no prose."""


def cache_key(slug: str, title: str) -> str:
    return hashlib.sha1(
        f"{slug}|{title.strip().lower()}|v{PROMPT_VERSION}".encode()).hexdigest()[:20]


def cached(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def validate(f: dict) -> dict:
    """Clamp/normalize one feature record; raise on structural garbage."""
    out = {
        "relevance": min(1.0, max(0.0, float(f["relevance"]))),
        "stance": f["stance"] if f["stance"] in STANCES else "neutral",
        "event_phase": f["event_phase"] if f["event_phase"] in PHASES else "none",
        "strength": min(1.0, max(0.0, float(f["strength"]))),
        "tone": min(1.0, max(-1.0, float(f.get("tone", 0.0)))),
        "event_type": f.get("event_type") if f.get("event_type") in EVENT_TYPES else "other",
        "entities": [str(e)[:60] for e in (f.get("entities") or [])][:5],
        "novelty": min(1.0, max(0.0, float(f.get("novelty", 0.5)))),
        "clarity": min(1.0, max(0.0, float(f.get("clarity", 0.5)))),
    }
    return out


def write_cache(key: str, features: dict, meta: dict | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(validate(features))
    if meta:
        rec["_meta"] = meta
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(rec, indent=1))


def _item_text(a: dict) -> str:
    """Article -> the capped text block the extractor sees (lede + last paragraph)."""
    parts = [a.get("title", "")]
    if a.get("trail"):
        parts.append(a["trail"])
    if a.get("lede"):
        parts.append(a["lede"])
    if a.get("last_para") and a.get("last_para") != a.get("lede"):
        parts.append(a["last_para"])
    return " || ".join(p.strip() for p in parts if p and p.strip())[:1200]


def pending_extractions(slug: str, question: str, criteria: str,
                        articles: list[dict]) -> list[dict]:
    """Work list of uncached (market, article) extractions for out-of-band processing."""
    out = []
    for a in articles:
        if not a.get("title"):
            continue
        key = cache_key(slug, a["title"])
        if cached(key) is not None:
            continue
        out.append({"cache_key": key, "slug": slug, "question": question,
                    "criteria": criteria[:500], "id": key,
                    "text": _item_text(a), "domain": a.get("domain", ""),
                    "seendate": a.get("seendate", "")})
    return out


def ingest_features(done: dict[str, dict], source: str) -> int:
    """Validate + cache out-of-band extraction results {cache_key: features}."""
    n = 0
    for key, feats in done.items():
        write_cache(key, feats, meta={"source": source})
        n += 1
    return n


def build_extraction_prompt(question: str, criteria: str, pending: list[dict]) -> str:
    items = "\n".join(f'- id={p["cache_key"]} [{p.get("domain","?")}] {p["text"]}'
                      for p in pending)
    return EXTRACTION_PROMPT.format(question=question, criteria=criteria[:500], items=items)


def _parse_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON array in reply: {text[:200]!r}")
    return json.loads(m.group(0))


def _match_rows(chunk: list[dict], rows: list[dict]) -> dict[str, dict]:
    """Model reply rows -> {cache_key: raw features}; tolerates order-only replies."""
    by_id = {r.get("id"): r for r in rows if isinstance(r, dict)}
    out = {}
    for idx, p in enumerate(chunk):
        r = by_id.get(p["cache_key"])
        if r is None and idx < len(rows):
            r = rows[idx]
        if r is not None:
            out[p["cache_key"]] = r
    return out


def _anthropic_rows(question: str, criteria: str, chunk: list[dict],
                    model: str) -> dict[str, dict]:
    key = os.environ.get(ANTHROPIC_KEY_ENV, "").strip()
    prompt = build_extraction_prompt(question, criteria, chunk)
    body = json.dumps({"model": model, "max_tokens": 4096,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    rows = _parse_array("".join(b.get("text", "") for b in resp.get("content", [])))
    return _match_rows(chunk, rows)


def _gemini_rows(question: str, criteria: str, chunk: list[dict],
                 model: str) -> dict[str, dict]:
    """Same prompt/output contract as the Anthropic path, via the Generative
    Language REST API (stdlib only)."""
    key = os.environ.get(GEMINI_KEY_ENV, "").strip()
    prompt = build_extraction_prompt(question, criteria, chunk)
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.0,
                                            "maxOutputTokens": 8192}}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, method="POST",
        headers={"x-goog-api-key": key, "content-type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    text = "".join(p.get("text", "")
                   for c in resp.get("candidates", [])[:1]
                   for p in (c.get("content") or {}).get("parts", []) or [])
    rows = _parse_array(text)
    return _match_rows(chunk, rows)


def pick_provider() -> str | None:
    """Extraction provider: explicit NEWSAGENT_EXTRACT_PROVIDER wins; otherwise
    auto — anthropic if its key is set, then gemini, else None (out-of-band)."""
    explicit = os.environ.get(PROVIDER_ENV, "").strip().lower()
    if explicit == "oob":
        return None
    if explicit in ("anthropic", "gemini"):
        return explicit
    if os.environ.get(ANTHROPIC_KEY_ENV, "").strip():
        return "anthropic"
    if os.environ.get(GEMINI_KEY_ENV, "").strip():
        return "gemini"
    return None


def extract_via_api(question: str, criteria: str, pending: list[dict],
                    model: str | None = None, provider: str = "anthropic") -> int:
    """API path: batched cheap-model extraction for one market's pending articles.

    provider "anthropic" (Haiku, the alpha-fit era default) or "gemini"
    (2.5 Flash, v3.1 flag — spot-check against the Haiku cache before trusting)."""
    if provider == "gemini":
        if not os.environ.get(GEMINI_KEY_ENV, "").strip():
            raise RuntimeError(f"{GEMINI_KEY_ENV} not set — Justin: export the free-tier "
                               "Gemini key, or use the out-of-band path.")
        model = model or GEMINI_MODEL
        rows_fn = _gemini_rows
    else:
        if not os.environ.get(ANTHROPIC_KEY_ENV, "").strip():
            raise RuntimeError(
                f"{ANTHROPIC_KEY_ENV} not set — use the out-of-band path: --stage extract "
                "writes extract_pending.json; ingest results with --features-file.")
        model = model or EXTRACT_MODEL
        rows_fn = _anthropic_rows
    n = 0
    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        for key2, raw in rows_fn(question, criteria, chunk, model).items():
            write_cache(key2, raw, meta={"source": f"api:{model}"})
            n += 1
    return n


def features_for(slug: str, articles: list[dict]) -> list[dict]:
    """Cached features joined back onto articles (order preserved; missing -> None)."""
    out = []
    for a in articles:
        key = cache_key(slug, a.get("title", ""))
        f = cached(key)
        out.append({"article": a, "cache_key": key, "features": f})
    return out
