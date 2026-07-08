"""Join-2c market selection — the pre-registered 5-screen politics-NegRisk filter.

Implements the per-market screens locked in
``polymarket/research/notes/market_making/mm_politics_negrisk_live_loop_design.md``
(Decision 1) for the operator's 1-contract live measurement run:

1. **negRisk: true** from the Gamma API row (the only reliable discriminator — never the
   Data-API ``negativeRisk`` field).
2. **Event bucket** ∈ {non-US elections, Trump personnel/policy, other politics, 2026 US
   races/midterms}; **2028 US presidential outrights are excluded** (no settled evidence +
   multi-year lockup). Keyword classifier below — transparent, and every candidate prints
   its bucket so the operator can override a misclassification.
3. **Non-top3 headroom ≥ 5%** in that specific market, from the corrected-carry cache
   (``mm_politics_negrisk_corrected_carry_recut_wallet_market.parquet``, keyed by the
   numeric Gamma market id). Markets with no history report ``UNKNOWN`` — the design's
   screen is historical, so new markets can't hard-pass it; they rank below known passes.
4. **Uninformed-flow preference**: the market's top-3 makers (by historical ``maker_usd``)
   joined to ``traders_directionality.parquet`` — prefer markets whose top-3 mean
   volume-weighted two-sided-directional share is BELOW the median across all politics
   top-3 wallets (retail-directional flow, not specialists). A preference, not a kill.
5. **Resolution clarity**: a scheduled ``endDate`` inside the horizon plus an
   objective-criterion keyword (election/confirmation/official …). Reported as a flag —
   the operator applies judgment (UMA-dispute risk is not machine-checkable).

Run from ``polymarket/research/``:

    PYTHONPATH=. uv run python scripts/mm_join2_market_screen.py            # table
    PYTHONPATH=. uv run python scripts/mm_join2_market_screen.py --emit-env 1

Read-only everywhere: public Gamma GETs + local parquet. No key, no order, no secret.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

GAMMA_URL = "https://gamma-api.polymarket.com"
# default urllib UA is 403-blocked on some PM surfaces — always send a browser-ish UA
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

CARRY_PARQUET = Path("data/analysis/mm_politics_negrisk_corrected_carry_recut_wallet_market.parquet")
DIRECTIONALITY_PARQUET = Path("data/directionality_classification/traders_directionality.parquet")
OUT_CSV = Path("data/analysis/csv_outputs/market_making/mm_join2_market_screen.csv")

HEADROOM_MIN = 0.05           # screen 3: ≥5% historical non-top3 fill share
DEFAULT_HORIZON_DAYS = 240    # screen 5: scheduled resolution inside this window
DEFAULT_MIN_VOLUME_USD = 10_000.0


# --------------------------------------------------------------------------------------
# screen 2 — bucket classifier (keyword-based, transparent)
# --------------------------------------------------------------------------------------

_NON_US_HINTS = (
    "uk|united kingdom|britain|british|canada|canadian|france|french|germany|german|"
    "ireland|irish|italy|italian|spain|spanish|portugal|netherlands|dutch|belgium|poland|"
    "polish|hungary|czech|slovak|romania|bulgaria|austria|swiss|switzerland|sweden|norway|"
    "denmark|finland|greece|turkey|turkish|japan|japanese|korea|korean|australia|"
    "new zealand|india|indian|brazil|brazilian|argentina|chile|chilean|colombia|peru|"
    "mexico|mexican|bolivia|ecuador|venezuela|israel|israeli|iran|iraq|egypt|nigeria|"
    "kenya|south africa|philippines|indonesia|malaysia|thailand|vietnam|taiwan|ukraine|"
    "moldova|georgia|armenia|serbia|kosovo|albania|scotland|wales|honduras|guatemala|"
    "costa rica|panama|paraguay|uruguay|dominican|haiti|jamaica"
)
_ELECTION_HINTS = (
    "election|elections|presidential|president|prime minister|premier|chancellor|"
    "parliament|parliamentary|mayor|mayoral|referendum|vote|runoff|run-off|primary|"
    "primaries|candidate|wins|winner|next leader|leadership"
)
_TRUMP_HINTS = (
    "trump|cabinet|secretary of|attorney general|fbi director|fed chair|nominee|"
    "nomination|confirm|confirmation|executive order|tariff|tariffs|deport|deportation|"
    "pardon|white house|administration official|czar|ambassador"
)
_POLITICS_HINTS = (
    "congress|senate|house of representatives|governor|impeach|supreme court|scotus|"
    "government shutdown|debt ceiling|bill passes|veto|minister|coalition|government|"
    "politician|political|geopolit|ceasefire|treaty|sanction|nato|resign|resignation|"
    "speaker of the house|midterm"
)

_RE_NON_US = re.compile(_NON_US_HINTS)
_RE_ELECTION = re.compile(_ELECTION_HINTS)
_RE_TRUMP = re.compile(_TRUMP_HINTS)
_RE_POLITICS = re.compile(_POLITICS_HINTS)
_RE_2028 = re.compile(r"2028")
_RE_US = re.compile(r"\bus\b|\bu\.s\.\b|united states|american|america\b")
_RE_2026_RACE = re.compile(r"2026.*(senate|house|governor|midterm|primary|race)|"
                           r"(senate|house|governor|midterm|primary|race).*2026")


def classify_bucket(text: str) -> tuple[str, bool]:
    """→ ``(bucket, in_scope)`` per Decision 1. ``text`` = question + slug + event title.

    Precedence: the 2028-outright EXCLUSION first (it would otherwise look like an
    election), then 2026 US races, non-US elections, Trump personnel/policy, other
    politics; anything without a politics signal is out of scope entirely.
    """
    t = text.lower()
    if _RE_2028.search(t) and (_RE_ELECTION.search(t) or "president" in t):
        return "us_2028_outrights", False          # excluded from Phase 2
    if _RE_2026_RACE.search(t) and _RE_US.search(t):
        return "us_2026_races", True
    if _RE_ELECTION.search(t) and _RE_NON_US.search(t) and not _RE_US.search(t):
        return "non_us_elections", True
    if _RE_TRUMP.search(t):
        return "trump_personnel_policy", True
    if _RE_POLITICS.search(t) or _RE_ELECTION.search(t):
        return "other_politics", True
    return "not_politics", False


# screen 5 — objective-criterion keywords (reported, operator judges)
_RE_OBJECTIVE = re.compile(
    r"election|elected|wins|winner|official|confirmed|confirmation|appointed|sworn in|"
    r"signs|passes|ratif|certif|inaugurat|becomes"
)


# --------------------------------------------------------------------------------------
# Gamma fetch (read-only, public)
# --------------------------------------------------------------------------------------

def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_active_negrisk_markets(
    gamma_url: str = GAMMA_URL,
    *,
    pages: int = 10,
    page_size: int = 100,
    get_json: Callable[[str], Any] = _http_get_json,
) -> list[dict[str, Any]]:
    """Active, order-accepting Gamma market rows with ``negRisk == true`` (screen 1)."""
    out: list[dict[str, Any]] = []
    for page in range(pages):
        qs = urllib.parse.urlencode({
            "closed": "false", "active": "true", "limit": page_size,
            "offset": page * page_size, "order": "volumeNum", "ascending": "false",
        })
        rows = get_json(f"{gamma_url}/markets?{qs}")
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("negRisk") is True and row.get("acceptingOrders", True):
                out.append(row)
        if len(rows) < page_size:
            break
    return out


# --------------------------------------------------------------------------------------
# screens 3+4 — historical caches (duckdb over parquet; read-only)
# --------------------------------------------------------------------------------------

def load_headroom_and_top3(
    carry_parquet: Path = CARRY_PARQUET,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Per historical market: non-top3 maker_usd share, and the top-3 maker addresses.

    "top-3" is ranked WITHIN the market by historical ``maker_usd`` (the accounting cache's
    per-wallet-market maker volume).
    """
    import duckdb

    con = duckdb.connect()
    df = con.execute(
        f"""
        WITH ranked AS (
            SELECT CAST(market_id AS VARCHAR) AS market_id,
                   address,
                   maker_usd,
                   row_number() OVER (PARTITION BY market_id ORDER BY maker_usd DESC) AS rk,
                   sum(maker_usd) OVER (PARTITION BY market_id) AS mkt_maker_usd
            FROM read_parquet('{carry_parquet.as_posix()}')
            WHERE maker_usd IS NOT NULL AND maker_usd > 0
        )
        SELECT market_id,
               max(mkt_maker_usd) AS mkt_maker_usd,
               sum(CASE WHEN rk > 3 THEN maker_usd ELSE 0 END) AS non_top3_usd,
               list(CASE WHEN rk <= 3 THEN address END) AS top3
        FROM ranked
        GROUP BY market_id
        """
    ).df()
    headroom: dict[str, float] = {}
    top3: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        mid = str(row["market_id"])
        total = float(row["mkt_maker_usd"]) if row["mkt_maker_usd"] else 0.0
        headroom[mid] = (float(row["non_top3_usd"]) / total) if total > 0 else 0.0
        top3_raw = row["top3"]           # duckdb list() → numpy array; truthiness is ambiguous
        top3[mid] = [a for a in (list(top3_raw) if top3_raw is not None else []) if a]
    return headroom, top3


def load_directional_scores(
    addresses: set[str],
    directionality_parquet: Path = DIRECTIONALITY_PARQUET,
) -> dict[str, float]:
    """Volume-weighted two-sided-directional share per wallet (screen 4's raw score)."""
    if not addresses:
        return {}
    import duckdb

    con = duckdb.connect()
    quoted = ",".join(f"'{a.lower()}'" for a in addresses)
    rows = con.execute(
        f"""
        SELECT lower(address) AS address, pct_markets_two_sided_directional_vw AS score
        FROM read_parquet('{directionality_parquet.as_posix()}')
        WHERE lower(address) IN ({quoted}) AND pct_markets_two_sided_directional_vw IS NOT NULL
        """
    ).fetchall()
    return {addr: float(score) for addr, score in rows}


# --------------------------------------------------------------------------------------
# the combined screen (pure — everything injected)
# --------------------------------------------------------------------------------------

@dataclass
class Candidate:
    condition_id: str
    gamma_id: str
    question: str
    bucket: str
    volume_usd: float
    end_date: str
    neg_risk: bool = True
    headroom: float | None = None            # None = no history (UNKNOWN)
    headroom_pass: bool | None = None
    top3_directional_score: float | None = None
    uninformed_flow_pref: bool | None = None
    resolution_clarity: bool = False
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "gamma_id": self.gamma_id,
            "question": self.question,
            "bucket": self.bucket,
            "volume_usd": round(self.volume_usd, 2),
            "end_date": self.end_date,
            "neg_risk": self.neg_risk,
            "headroom_non_top3": "" if self.headroom is None else round(self.headroom, 4),
            "headroom_pass": "UNKNOWN" if self.headroom_pass is None else self.headroom_pass,
            "top3_directional_score": (
                "" if self.top3_directional_score is None
                else round(self.top3_directional_score, 4)
            ),
            "uninformed_flow_pref": (
                "UNKNOWN" if self.uninformed_flow_pref is None else self.uninformed_flow_pref
            ),
            "resolution_clarity": self.resolution_clarity,
            "notes": "; ".join(self.notes),
        }


def _fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def screen_markets(
    gamma_rows: list[dict[str, Any]],
    headroom: dict[str, float],
    top3: dict[str, list[str]],
    directional_scores: dict[str, float],
    *,
    min_volume_usd: float = DEFAULT_MIN_VOLUME_USD,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    now_utc=None,
) -> list[Candidate]:
    """Apply screens 2–5 to (already screen-1-filtered) Gamma rows; rank the survivors.

    Hard screens: bucket in scope (2), volume floor (liquidity sanity). Screen 3 is hard
    when history exists (≥5% non-top3), UNKNOWN when it doesn't. Screens 4–5 are ranked
    preferences, reported per candidate. Sort order: screen-3 known-pass first, then
    uninformed-flow preference, then resolution clarity, then volume.
    """
    from datetime import datetime, timedelta, timezone

    now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    universe_scores = sorted(directional_scores.values())
    median_score = (
        universe_scores[len(universe_scores) // 2] if universe_scores else None
    )

    out: list[Candidate] = []
    for row in gamma_rows:
        cond = str(row.get("conditionId") or "").lower()
        gid = str(row.get("id") or "")
        question = str(row.get("question") or "")
        slug = str(row.get("slug") or "")
        event_title = ""
        events = row.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            event_title = str(events[0].get("title") or "")
        text = " ".join([question, slug, event_title])

        bucket, in_scope = classify_bucket(text)
        if not cond or not in_scope:
            continue
        volume = _fnum(row.get("volumeNum") or row.get("volume"))
        if volume < min_volume_usd:
            continue

        cand = Candidate(
            condition_id=cond, gamma_id=gid, question=question, bucket=bucket,
            volume_usd=volume, end_date=str(row.get("endDate") or ""),
        )

        # screen 3 — historical non-top3 headroom (hard when known)
        hr = headroom.get(gid)
        if hr is None:
            cand.headroom_pass = None
            cand.notes.append("no history in corrected-carry cache — screen 3 UNKNOWN")
        else:
            cand.headroom = hr
            cand.headroom_pass = hr >= HEADROOM_MIN
            if not cand.headroom_pass:
                cand.notes.append(f"non-top3 share {hr:.1%} < {HEADROOM_MIN:.0%} — screen 3 FAIL")

        # screen 4 — top-3 directionality (preference)
        addrs = [a.lower() for a in top3.get(gid, [])]
        scores = [directional_scores[a] for a in addrs if a in directional_scores]
        if scores and median_score is not None:
            cand.top3_directional_score = sum(scores) / len(scores)
            cand.uninformed_flow_pref = cand.top3_directional_score < median_score
        else:
            cand.notes.append("top-3 makers unscored in directionality cache — screen 4 UNKNOWN")

        # screen 5 — resolution clarity (scheduled end date + objective keyword)
        end_ok = False
        end_raw = cand.end_date
        if end_raw:
            try:
                end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                end_ok = now < end_dt <= now + timedelta(days=horizon_days)
            except ValueError:
                cand.notes.append(f"unparseable endDate {end_raw!r}")
        cand.resolution_clarity = bool(end_ok and _RE_OBJECTIVE.search(text.lower()))
        if not end_ok:
            cand.notes.append(f"endDate outside {horizon_days:.0f}d horizon (or missing)")

        if cand.headroom_pass is False:
            continue        # hard fail only when history exists and says < 5%
        out.append(cand)

    def sort_key(c: Candidate):
        return (
            0 if c.headroom_pass is True else 1,            # known-pass first
            0 if c.uninformed_flow_pref is True else 1,     # uninformed flow preferred
            0 if c.resolution_clarity else 1,
            -c.volume_usd,
        )

    out.sort(key=sort_key)
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def render_table(cands: list[Candidate], top: int = 15) -> str:
    lines = [
        f"{'#':>2} {'bucket':<24} {'hdrm':>7} {'dir':>7} {'clar':>5} "
        f"{'volume':>12} {'end':<12} question",
    ]
    for i, c in enumerate(cands[:top], 1):
        hr = "UNK" if c.headroom is None else f"{c.headroom:.1%}"
        ds = "UNK" if c.top3_directional_score is None else f"{c.top3_directional_score:.2f}"
        lines.append(
            f"{i:>2} {c.bucket:<24} {hr:>7} {ds:>7} {str(c.resolution_clarity):>5} "
            f"{c.volume_usd:>12,.0f} {c.end_date[:10]:<12} {c.question[:60]}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages", type=int, default=10, help="Gamma pages (100 rows each)")
    ap.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME_USD)
    ap.add_argument("--horizon-days", type=float, default=DEFAULT_HORIZON_DAYS)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--emit-env", type=int, metavar="N",
                    help="print the env lines for the N-th ranked candidate")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)

    print("[screen] fetching active NegRisk markets from Gamma (read-only)...", file=sys.stderr)
    rows = fetch_active_negrisk_markets(pages=args.pages)
    print(f"[screen] gamma rows with negRisk=true: {len(rows)}", file=sys.stderr)

    headroom, top3 = load_headroom_and_top3()
    all_top3 = {a.lower() for addrs in top3.values() for a in addrs}
    scores = load_directional_scores(all_top3)

    cands = screen_markets(
        rows, headroom, top3, scores,
        min_volume_usd=args.min_volume, horizon_days=args.horizon_days,
    )
    print(render_table(cands, top=args.top))
    print(f"[screen] candidates: {len(cands)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        if cands:
            writer = csv.DictWriter(fh, fieldnames=list(cands[0].as_row().keys()))
            writer.writeheader()
            for c in cands:
                writer.writerow(c.as_row())
    print(f"[screen] written: {args.out}", file=sys.stderr)

    if args.emit_env:
        idx = args.emit_env - 1
        if not (0 <= idx < len(cands)):
            print(f"[screen] --emit-env {args.emit_env} out of range", file=sys.stderr)
            return 2
        chosen = cands[idx]
        print()
        print("# Join-2c selected market — paste into the run env (see the runbook):")
        print(f"POLYMARKET_MAKER_CONDITION_ID={chosen.condition_id}")
        print(f"# {chosen.question} | bucket={chosen.bucket} | end={chosen.end_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
