"""GDELT GKG historical tone/volume via Google BigQuery (NOW BUILDING: v2.1 feature).

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

SERIES_CACHE = DATA / "gdelt_daily.json"

# Cost discipline: ONE scan per date range covers ALL markets (COUNTIF per name-set
# over the same DATE/AllNames/V2Tone columns) — dry-run measured ~3.4 GB for a
# 35-day window, ~0.3% of the 1 TB/mo free tier. Name keys are AND-substring
# matches against LOWER(AllNames); several markets in one event family may share
# an attention series — acceptable and declared, because the burst feature is a
# per-market z-score against its own trailing baseline (relative, not absolute).


def _match_expr(keys: list[str]) -> str:
    parts = " AND ".join(
        "names LIKE '%" + k.lower().replace("'", "").replace("%", "") + "%'"
        for k in keys)
    return f"({parts})"


def build_multi_market_sql(name_keys: dict[str, list[str]], start: str, end: str) -> str:
    """One partitioned scan, per-day volume + mean tone per market name-set."""
    cols = []
    for i, (slug, keys) in enumerate(sorted(name_keys.items())):
        m = _match_expr(keys)
        cols.append(f"COUNTIF({m}) AS n_{i},\n  AVG(IF({m}, tone, NULL)) AS tone_{i}")
    cols_sql = ",\n  ".join(cols)
    return f"""
SELECT SUBSTR(CAST(DATE AS STRING), 1, 8) AS day,
  {cols_sql}
FROM (
  SELECT DATE, LOWER(AllNames) AS names,
         CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')
)
GROUP BY day ORDER BY day
"""


def load_series() -> dict:
    return json.loads(SERIES_CACHE.read_text()) if SERIES_CACHE.exists() else {}


def pull_daily_series(name_keys: dict[str, list[str]], start: str, end: str,
                      max_gb: float = 25.0) -> dict:
    """Pull per-day {n, tone} per market and merge into the local cache.

    Dry-runs first and refuses to run past max_gb (free-tier guard). Returns the
    merged cache {slug: {YYYYMMDD: {n, tone}}}."""
    ok, why = bigquery_available()
    if not ok:
        raise RuntimeError(f"GDELT BigQuery path unavailable: {why}")
    from google.cloud import bigquery
    client = bigquery.Client()
    sql = build_multi_market_sql(name_keys, start, end)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    gb = dry.total_bytes_processed / 1e9
    if gb > max_gb:
        raise RuntimeError(f"query would scan {gb:.1f} GB > guard {max_gb} GB — "
                           "narrow the date range")
    rows = list(client.query(sql).result())
    slugs = sorted(name_keys)
    cache = load_series()
    for r in rows:
        for i, slug in enumerate(slugs):
            n = getattr(r, f"n_{i}")
            tone = getattr(r, f"tone_{i}")
            cache.setdefault(slug, {})[r.day] = {
                "n": int(n), "tone": round(float(tone), 3) if tone is not None else None}
    SERIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SERIES_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def burst_z(series: dict[str, dict], day: str, trailing: int = 14,
            z_clip: float = 3.0) -> dict | None:
    """Attention-burst feature for one market-day from its own daily series.

    vol_z: z-score of the day's matched-article count vs the trailing `trailing`
    days (needs >= 5 of them; else None). tone / tone_shift reported for display
    and future calibration — NOT wired into the FV direction (tone->direction is
    question-specific; see findings note)."""
    from datetime import datetime, timedelta
    d0 = datetime.strptime(day, "%Y%m%d")
    today = series.get(day)
    if today is None:
        return None
    hist = []
    for k in range(1, trailing + 1):
        rec = series.get((d0 - timedelta(days=k)).strftime("%Y%m%d"))
        if rec is not None:
            hist.append(rec)
    if len(hist) < 5:
        return None
    ns = [h["n"] for h in hist]
    mean = sum(ns) / len(ns)
    var = sum((x - mean) ** 2 for x in ns) / len(ns)
    sd = max(var ** 0.5, 1.0, 0.1 * mean)   # floor: tiny/quiet series can't fake a burst
    vol_z = max(-z_clip, min(z_clip, (today["n"] - mean) / sd))
    tones = [h["tone"] for h in hist if h["tone"] is not None]
    tone_base = sum(tones) / len(tones) if tones else None
    tone_shift = (round(today["tone"] - tone_base, 3)
                  if today["tone"] is not None and tone_base is not None else None)
    return {"n": today["n"], "n_trailing_mean": round(mean, 1),
            "vol_z": round(vol_z, 3), "tone": today["tone"], "tone_shift": tone_shift}


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
