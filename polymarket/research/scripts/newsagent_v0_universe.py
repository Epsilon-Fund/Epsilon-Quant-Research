"""News-agent v0 gate: build the resolved-market universe per the pre-registered rule.

Pre-registration: notes/news_agent/newsagent_v0_gate_preregistration.md
Rule: Gamma politics tag_id=2, closed, endDate in [2026-06-05, 2026-07-03], volumeNum >= $3M,
binary, mid in [5c,95c] on >= 3 snapshot dates while still trading, <= 2 markets per
event family, target n=10, >= 5 families, >= 1 UK family if any passes.

Outputs (append-only raw cache + result CSV):
  data/newsagent/v0/prices/<slug>.json         hourly YES mid history, chunked fetch
  data/analysis/csv_outputs/news_agent/newsagent_v0_universe.csv
  data/newsagent/v0/universe_selected.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
PRICES = RAW / "prices"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]
SNAPSHOT_HOUR_UTC = 12

# Transparent keyword -> family map (first match wins, checked in order).
FAMILY_RULES = [
    ("fed_june", ["fed interest rates"]),
    ("iran_regime", ["iranian regime", "khamenei", "iran leadership change"]),
    ("iran_us_deal", ["permanent peace deal", "us and iran sign", "us-iran nuclear deal",
                       "end enrichment of uranium", "surrender enriched uranium",
                       "text of the us-iran agreement", "diplomatic meeting",
                       "agreement/ceasefire extension", "obtains iranian enriched uranium",
                       "us x iran ceasefire"]),
    ("iran_airspace", ["closes its airspace"]),
    ("iran_war_misc", ["strikes iran", "conflict ends", "military operations against iran",
                        "netanyahu enter iran", "withdraw troops from the iranian",
                        "strait of hormuz", "transit fees"]),
    ("starmer_uk", ["starmer"]),
    ("colombia_election", ["colombian presidential"]),
    ("peru_election", ["peruvian presidential"]),
    ("netanyahu_out", ["netanyahu out"]),
    ("trump_out", ["trump out as president"]),
    ("serbia_vucic", ["vučić", "serbian president"]),
    ("russia_ukraine", ["russia x ukraine"]),
    ("russia_nuke", ["russia test a nuclear weapon"]),
    ("putin_out", ["putin out"]),
    ("taiwan", ["china invade taiwan"]),
    ("project_freedom", ["project freedom"]),
    ("aliens", ["aliens exist"]),
]
UK_FAMILIES = {"starmer_uk"}


def http_get(url: str, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))
    raise RuntimeError("unreachable")


def family_of(question: str) -> str:
    q = question.lower()
    for fam, keys in FAMILY_RULES:
        if any(k in q for k in keys):
            return fam
    return "other"


def safe_name(slug: str) -> str:
    """Filesystem-safe cache key: long Gamma slugs exceed macOS's 255-byte name limit."""
    if len(slug) <= 100:
        return slug
    return slug[:80] + "-" + hashlib.sha1(slug.encode()).hexdigest()[:10]


def fetch_price_history(slug: str, token: str, start: datetime, end: datetime) -> list[dict]:
    """Chunked <=9d fetches (CLOB /prices-history has a silent ~15d span cap). Cached append-only."""
    cache = PRICES / f"{safe_name(slug)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    points: list[dict] = []
    cur = start
    while cur < end:
        chunk_end = min(datetime.fromtimestamp(cur.timestamp() + 9 * 86400, tz=timezone.utc), end)
        url = (f"https://clob.polymarket.com/prices-history?market={token}"
               f"&startTs={int(cur.timestamp())}&endTs={int(chunk_end.timestamp())}&fidelity=60")
        try:
            points.extend(json.loads(http_get(url)).get("history", []))
        except Exception as exc:  # market may have no book in a chunk; keep going
            print(f"    warn: {slug} chunk {cur.date()} failed: {exc}")
        cur = chunk_end
        time.sleep(0.25)
    seen: dict[int, float] = {}
    for p in points:
        seen[int(p["t"])] = float(p["p"])
    out = [{"t": t, "p": seen[t]} for t in sorted(seen)]
    cache.write_text(json.dumps(out))
    return out


def snapshot_mid(history: list[dict], snap_ts: int, max_stale_h: int = 48) -> float | None:
    """Last price at or before snap_ts, no older than max_stale_h hours (lookahead-free)."""
    best = None
    for p in history:
        if p["t"] <= snap_ts:
            best = p
        else:
            break
    if best is None or snap_ts - best["t"] > max_stale_h * 3600:
        return None
    return best["p"]


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    candidates = json.loads((RAW / "candidates_raw.json").read_text())
    start = datetime(2026, 6, 5, tzinfo=timezone.utc)
    end = datetime(2026, 7, 2, tzinfo=timezone.utc)

    rows = []
    for c in candidates:
        outcomes = json.loads(c["outcomes"]) if isinstance(c["outcomes"], str) else c["outcomes"]
        if outcomes != ["Yes", "No"]:
            continue  # binary only
        token_yes = (json.loads(c["clobTokenIds"]) if isinstance(c["clobTokenIds"], str)
                     else c["clobTokenIds"])[0]
        closed_ts = datetime.fromisoformat(c["closedTime"].replace(" ", "T").replace("+00", "+00:00")).timestamp()
        prices = json.loads(c["outcomePrices"]) if isinstance(c["outcomePrices"], str) else c["outcomePrices"]
        outcome_yes = int(float(prices[0]) > 0.5)
        print(f"  {c['slug'][:60]} ...")
        hist = fetch_price_history(c["slug"], token_yes, start, end)
        snaps: dict[str, float | None] = {}
        n_informative = 0
        for d in SNAPSHOT_DATES:
            ts = int(datetime.fromisoformat(f"{d}T{SNAPSHOT_HOUR_UTC:02d}:00:00+00:00").timestamp())
            if ts >= closed_ts:
                snaps[d] = None  # market no longer trading
                continue
            mid = snapshot_mid(hist, ts)
            snaps[d] = mid
            if mid is not None and 0.05 <= mid <= 0.95:
                n_informative += 1
        rows.append({
            "slug": c["slug"], "question": c["question"], "family": family_of(c["question"]),
            "volume_usd": c["volumeNum"], "end_date": c["endDate"][:10],
            "closed_time": c["closedTime"][:19], "outcome_yes": outcome_yes,
            "n_informative_snapshots": n_informative,
            "passes_info_filter": n_informative >= 3,
            **{f"mid_{d}": (round(snaps[d], 3) if snaps[d] is not None else "") for d in SNAPSHOT_DATES},
        })

    # Selection: volume-ranked, <=2 per family, only info-filter passers, target 10.
    selected, fam_count = [], {}
    for r in sorted(rows, key=lambda r: -r["volume_usd"]):
        if not r["passes_info_filter"]:
            continue
        if fam_count.get(r["family"], 0) >= 2:
            continue
        fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
        selected.append(r)
        if len(selected) >= 10:
            break
    for r in rows:
        r["selected"] = r["slug"] in {s["slug"] for s in selected}

    with open(CSV_OUT / "newsagent_v0_universe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["volume_usd"]))
    (RAW / "universe_selected.json").write_text(json.dumps(selected, indent=1))

    print(f"\n{len(rows)} binary candidates; {sum(r['passes_info_filter'] for r in rows)} pass info filter; "
          f"{len(selected)} selected across {len(fam_count)} families "
          f"(UK included: {bool(set(fam_count) & UK_FAMILIES)})")
    for s in selected:
        print(f"  [{s['family']:<18}] ${s['volume_usd']:>11,}  YES={s['outcome_yes']}  {s['question'][:60]}")


if __name__ == "__main__":
    main()
