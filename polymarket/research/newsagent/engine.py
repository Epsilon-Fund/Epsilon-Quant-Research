"""Forecast engine: the v0b five-perspective protocol as a reusable component.

The prompt and aggregation are byte-compatible with the v0b gate (Amendment 2 in
newsagent_v0_gate_preregistration): five perspective-diverse estimates, trimmed
mean computed HERE (drop min & max, average middle 3), 80% band = mid-3 spread
floored at +/-8pp, clipped [1, 99]. The forecaster never sees the market mid.

Forecast execution paths:
  1. Anthropic API (ANTHROPIC_API_KEY set) — one message per market, stdlib HTTP.
  2. --forecasts-file — a JSON of raw {key: {estimates_pct, drivers, decisive_evidence}}
     produced out-of-band (e.g. by Claude Code subagents); the engine only aggregates.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone

from .config import ANTHROPIC_KEY_ENV

PROMPT_TEMPLATE = """You are a calibrated superforecaster producing an INDEPENDENT fair-value probability for a prediction-market question. Today is {date}. Rely ONLY on the evidence packet below plus general knowledge of how the world works. You have no market price and must not guess one; produce your own view.

QUESTION: {question}

RESOLUTION CRITERIA: {description}

DEADLINE: {deadline} ({days_left} days from today)

EVIDENCE PACKET (news headlines retrieved as of today, most relevant first):
{packet}

Method — produce FIVE independent estimates of P(YES), each from a different angle. Reason privately; report all five numbers:
1. BASE-RATE: the outside-view base rate for this class of event resolving YES within {days_left} days, ignoring the packet entirely.
2. EVIDENCE-FORWARD: take the packet at face value. If credible reporting indicates an imminent, in-progress, or completed qualifying event, this estimate MUST move materially toward it (cite the driving headline in drivers). Distinguish confirmed fact from speculation — but do not dismiss a strong, specific signal merely because the event has not completed yet.
3. SKEPTIC: what would have to be true for the packet headlines to mislead? Estimate under that reading.
4. REFERENCE-CLASS: the closest concrete historical analogs to this exact situation (not the generic class); estimate from their outcomes.
5. ADVERSARIAL: assume a smart, informed person disagrees with your evidence-forward number; steelman their case, then give your all-things-considered estimate.

Set "decisive_evidence" to true only if the packet contains credible reporting that the qualifying event is imminent, in progress, or completed.

Reply with a single JSON object only, no prose:
{{"estimates_pct": [<e1>, <e2>, <e3>, <e4>, <e5>], "drivers": ["<up to 3 short strings citing packet items or base-rate logic>"], "decisive_evidence": <true|false>}}
"""


def build_prompt(market: dict, packet: dict, asof: datetime | None = None) -> str:
    t = asof or datetime.now(timezone.utc)
    deadline = market["end_date"]
    days_left = 0
    if deadline:
        days_left = max(0, (datetime.fromisoformat(deadline.replace("Z", "+00:00")) - t).days)
    lines = "\n".join(f"- [{a.get('seendate','?')}] {a.get('domain','?')}: {a['title']}"
                      for a in packet["articles"]) or "(no articles retrieved)"
    return PROMPT_TEMPLATE.format(date=t.strftime("%Y-%m-%d"), question=market["question"],
                                  description=market["description"], deadline=deadline[:10],
                                  days_left=days_left, packet=lines)


def aggregate(estimates: list[float]) -> dict:
    """Trimmed mean + floored band, identical to the v0b gate aggregation."""
    assert len(estimates) == 5, f"need 5 estimates, got {len(estimates)}"
    mid3 = sorted(float(e) for e in estimates)[1:4]
    p = sum(mid3) / 3.0
    half = max(8.0, (mid3[2] - mid3[0]) / 2.0)
    return {"p_pct": round(p, 1),
            "band_lo_pct": round(max(1.0, p - half), 1),
            "band_hi_pct": round(min(99.0, p + half), 1)}


def parse_reply(text: str) -> dict:
    """Extract the JSON object from a model reply (tolerates surrounding prose/fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    raw = json.loads(m.group(0))
    assert len(raw["estimates_pct"]) == 5
    return raw


def anthropic_forecast(prompt: str, model: str = "claude-sonnet-5") -> dict:
    """One forecast via the Anthropic Messages API (stdlib only). Requires the API key."""
    key = os.environ.get(ANTHROPIC_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{ANTHROPIC_KEY_ENV} is not set. Either export a real key or run the "
            "forecast stage out-of-band and pass --forecasts-file (see run_daily.py).")
    body = json.dumps({"model": model, "max_tokens": 1024,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return parse_reply("".join(b.get("text", "") for b in resp.get("content", [])))
