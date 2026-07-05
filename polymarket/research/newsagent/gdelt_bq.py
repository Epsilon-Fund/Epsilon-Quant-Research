"""GDELT GKG historical tone/volume via Google BigQuery (STRETCH: deep calibration).

Why BigQuery: the GDELT DOC 2.0 API sits behind an aggressive datacenter-IP
throttle (persistent 429s reproduced from two unrelated IPs, incl. the Hetzner
VPS — see newsagent_v0_gate_findings Amendment 1). `gdelt-bq.gdeltv2.gkg` on
BigQuery bypasses the API entirely: free tier 1 TB scanned/month, full history.

NEEDS JUSTIN (one-time):
  1. GCP project -> enable the BigQuery API.
  2. Service account with roles BigQuery User + BigQuery Job User.
  3. Download the JSON key; export GOOGLE_APPLICATION_CREDENTIALS=/path/key.json
  4. `uv add google-cloud-bigquery` in polymarket/research (not added by default —
     no dependency lands until the credential exists).

Fallback (no GCP): `doc_api_fallback()` — DOC 2.0 from a RESIDENTIAL IP only
(this machine, not the VPS), long backoff + jitter, append-only cache. Client-side
`seendate` filtering is mandatory: the API's end-bound silently leaks ~24h of
future articles (radar-verified) — never trust the server-side window for
lookahead-free reconstruction.

Live/current news stays on Guardian + RSS (feeds.py); this module is only for
HISTORICAL calibration enrichment (V2Tone per query/day as an optional Stage-B
feature).
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime

from .config import DATA

CACHE = DATA / "gdelt_cache"

GKG_TONE_SQL = """
-- Daily mean V2Tone + article volume for a theme/keyword window (partitioned scan).
SELECT
  SUBSTR(CAST(DATE AS STRING), 1, 8) AS day,
  COUNT(*) AS n_articles,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS mean_tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start) AND TIMESTAMP(@end)
  AND LOWER(DocumentIdentifier) LIKE @like_pattern
GROUP BY day ORDER BY day
"""


def bigquery_available() -> tuple[bool, str]:
    """Report whether the BigQuery path can run, with the exact missing step."""
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not cred:
        return False, ("GOOGLE_APPLICATION_CREDENTIALS not set — Justin: GCP project + "
                       "BigQuery API + service-account JSON key (see module docstring)")
    if not os.path.exists(cred):
        return False, f"credential file missing: {cred}"
    try:
        import google.cloud.bigquery  # noqa: F401
    except ImportError:
        return False, ("google-cloud-bigquery not installed — run "
                       "`uv add google-cloud-bigquery` in polymarket/research")
    return True, "ready"


def daily_tone(like_pattern: str, start: str, end: str) -> list[dict]:
    """Daily (n_articles, mean_tone) for URLs matching like_pattern, via BigQuery."""
    ok, why = bigquery_available()
    if not ok:
        raise RuntimeError(f"GDELT BigQuery path unavailable: {why}")
    from google.cloud import bigquery
    client = bigquery.Client()
    job = client.query(GKG_TONE_SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start", "STRING", start),
            bigquery.ScalarQueryParameter("end", "STRING", end),
            bigquery.ScalarQueryParameter("like_pattern", "STRING", like_pattern),
        ]))
    return [{"day": r.day, "n_articles": r.n_articles,
             "mean_tone": round(r.mean_tone, 3)} for r in job.result()]


def doc_api_fallback(query: str, start: datetime, end: datetime,
                     max_records: int = 75) -> list[dict]:
    """DOC 2.0 fallback — residential IP only, long backoff, cached append-only.

    Applies the mandatory client-side seendate filter (server end-bound leaks ~24h)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{query}|{start:%Y%m%d}|{end:%Y%m%d}".replace("/", "_")
    cache = CACHE / (str(abs(hash(key)))[:16] + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    params = {"query": query, "mode": "artlist", "format": "json",
              "maxrecords": str(max_records), "sort": "hybridrel",
              "startdatetime": start.strftime("%Y%m%d%H%M%S"),
              "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            raw = json.loads(urllib.request.urlopen(req, timeout=45).read())
            arts = []
            cutoff = end.strftime("%Y%m%dT%H%M%SZ")
            for a in raw.get("articles", []):
                if a.get("seendate", "") > cutoff:   # mandatory lookahead guard
                    continue
                arts.append({"title": a.get("title", ""), "seendate": a.get("seendate", ""),
                             "domain": a.get("domain", ""), "url": a.get("url", "")})
            cache.write_text(json.dumps(arts))
            return arts
        except Exception as e:  # 429s: long jittered backoff, then give up loudly
            last_err = e
            time.sleep(30 * (attempt + 1) + random.uniform(0, 10))
    raise RuntimeError(f"GDELT DOC API unreachable after backoff ({last_err}); "
                       "use the BigQuery path or Guardian/RSS")
