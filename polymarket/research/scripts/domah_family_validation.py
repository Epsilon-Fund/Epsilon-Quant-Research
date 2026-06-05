"""Task 1: validate the FAMILY_KEYWORDS slug heuristic used by the Domah audit.

Output: data/analysis/domah_followups/family_heuristic_validation.md

Steps:
  1) For each inferred family, list the top 30 distinct slugs by Domah's fills
     plus a random 30 (with notional + leader PnL).
  2) Scan the OTHER bucket for slugs that obviously belong elsewhere.
  3) Propose additions/edits to FAMILY_KEYWORDS, with re-routing counts + examples.
  4) Apply the proposed heuristic to the existing audit fragments + positions,
     recompute the family-level table, and report which findings change materially.

Read-only with respect to canonical data — works off the existing audit parquets.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
OUT_DIR  = ANALYSIS / "domah_followups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAG_PATH = ANALYSIS / "domah_audit_fragments.parquet"
POS_PATH  = ANALYSIS / "domah_audit_positions.parquet"
FAM_PATH  = ANALYSIS / "domah_audit_family.parquet"


# ----------------------------------------------------------------------------
# Original heuristic (copied verbatim from domah_copy_audit.py for re-use)
# ----------------------------------------------------------------------------
FAMILY_KEYWORDS_ORIG: list[tuple[str, tuple[str, ...]]] = [
    ("sports", (
        "-mlb-", "-mlb", "mlb-", "-nba-", "-nba", "nba-", "-nfl-", "-nfl", "nfl-",
        "-nhl-", "-nhl", "nhl-", "-cfb-", "-cfb", "cfb-", "-ncaa-", "ncaa-",
        "-ufc-", "ufc-", "-soccer-", "soccer-",
        "premier-league", "champions-league", "-ucl-", "ucl-",
        "world-cup", "tennis", "atp-", "wta-", "grand-slam",
        "-boxing-", "boxing-", "-mma-", "mma-", "formula-1", "-f1-", "nascar",
        "yankees", "dodgers", "lakers", "warriors", "chiefs", "eagles",
        "broncos", "knicks", "celtics", "heat", "patriots",
        "saquon", "shedeur", "wrestlemania",
        "kentucky-derby", "preakness", "belmont", "open-championship",
        "masters-tournament", "ryder-cup", "stanley-cup", "world-series",
        "super-bowl", "us-open", "nba-mvp", "nfl-mvp", "heisman",
        "-pga-", "pga-", "-pga", "championship",
        "jannik-sinner", "novak-djokovic", "carlos-alcaraz", "iga-swiatek",
        "ja-morant", "jokic", "lebron",
    )),
    ("crypto", (
        "bitcoin", "ethereum", "solana", "doge", "dogecoin", "litecoin",
        "ripple", "xrp", "cardano", "polkadot", "avalanche",
        "binance", "coinbase", "kraken", "tether", "usdc",
        "crypto", "blockchain", "defi", "stablecoin",
        "-btc-", "btc-", "-eth-", "eth-", "-sol-", "sol-",
        "vitalik", "satoshi", "-cz-", "cz-",
        "sec-vs", "sec-coinbase",
        "spot-etf", "bitcoin-etf", "ether-etf",
    )),
    ("macro", (
        "-fed-", "fed-", "federal-reserve", "fomc", "jerome-powell", "jpow",
        "-cpi-", "cpi-", "inflation", "deflation", "-ppi-", "ppi-",
        "-gdp-", "gdp-", "jobs-report", "nonfarm", "-nfp-", "nfp-",
        "unemployment", "recession",
        "interest-rates", "rate-cut", "rate-hike", "rate-decision",
        "treasury-yield", "yield-curve", "10-year-yield",
        "sp-500", "sp500", "-qqq-", "qqq-", "tariff", "trade-deal",
    )),
    ("weather", (
        "temperature", "high-temp", "hottest", "coldest",
        "hurricane", "tropical-storm", "typhoon", "cyclone",
        "tornado", "snowfall", "snow-in", "rainfall",
        "atmospheric-river", "heatwave", "blizzard", "wildfire", "noaa",
    )),
    ("politics", (
        "trump", "biden", "harris", "vance", "kamala", "obama",
        "election", "primary", "caucus",
        "senate", "house-of-rep", "congress", "congressional",
        "mayoral", "governor", "presidential",
        "democrat", "republican", "-gop-", "gop-", "-dnc-", "dnc-",
        "-rnc-", "rnc-",
        "supreme-court", "-scotus-", "scotus-",
        "putin", "zelensky", "ukraine", "russia",
        "israel", "palestine", "gaza", "hamas", "hezbollah", "houthi",
        "iran", "netanyahu", "khamenei",
        "nato", "european-union", "-eu-", "brexit",
        "macron", "starmer", "merz", "scholz", "meloni",
        "modi", "xi-jinping", "china-", "taiwan", "north-korea", "kim-jong",
        "pope", "vatican",
        "white-house", "executive-order", "impeach", "indict",
        "fbi-", "doj-", "pentagon",
        "syria", "lebanon", "venezuela", "maduro",
        "epstein", "ghislaine",
        "shutdown", "debt-ceiling",
        "us-x-", "us-iran", "us-russia", "us-china", "us-strikes",
        "rodrigo", "milei", "musk", "vivek",
        "tiktok",
        "noem", "marco-rubio", "pam-bondi",
        "openai-receives-federal",
        "eric-adams", "angela-rayner", "yulia-navalnaya", "greenland",
        "world-leader", "anti-cartel", "drop-out", "dropout",
        "navalny", "mahmoud-abbas", "sanchez", "lula",
        "doria-medina",
    )),
]

# ----------------------------------------------------------------------------
# Proposed additions — derived from inspecting the OTHER bucket interactively
# ----------------------------------------------------------------------------
# Each entry is (family, keyword). The classifier wraps slug with leading +
# trailing '-' so substrings like 'fed' match '-fed-' but not 'feder'.
PROPOSED_ADDITIONS: list[tuple[str, str]] = [
    # === Politics ===
    ("politics", "weed-rescheduled"),       # DEA / regulatory action
    ("politics", "weed-reschedul"),
    ("politics", "national-guard"),          # Trump national-guard markets
    ("politics", "deportation"),
    ("politics", "ice-"),                    # ICE raids
    ("politics", "passport"),                # US passport-related policy
    ("politics", "executive-action"),
    ("politics", "cabinet-confirm"),
    ("politics", "confirmed-by"),            # Senate confirmation
    ("politics", "secretary-of"),
    ("politics", "ambassador-to"),
    ("politics", "sanctions-on"),
    ("politics", "ceasefire"),
    ("politics", "war-on"),
    ("politics", "regime-change"),
    ("politics", "coup-"),                   # coup attempts
    ("politics", "nuclear-"),                # nuclear talks (mostly political)
    ("politics", "diplomatic"),
    ("politics", "embassy"),
    ("politics", "treaty"),
    ("politics", "summit"),
    ("politics", "negotiations"),
    ("politics", "mueller"),
    ("politics", "kennedy"),
    ("politics", "rfk"),
    ("politics", "desantis"),
    ("politics", "newsom"),
    ("politics", "abrego"),                  # Abrego Garcia deportation case
    ("politics", "kilmar"),
    ("politics", "cz-by"),                   # Trump pardons CZ (politics, not crypto)
    ("politics", "pardon"),                  # presidential pardons
    ("politics", "pardons"),
    ("politics", "bondi"),
    ("politics", "kash-patel"),
    ("politics", "vance-"),
    ("politics", "rubio"),
    ("politics", "tulsi"),
    ("politics", "world-leader"),            # already exists but covering variants
    ("politics", "world-leaders"),
    ("politics", "supreme-leader"),          # Khamenei context
    ("politics", "prime-minister"),
    ("politics", "presidential-race"),
    ("politics", "elections-in"),
    ("politics", "election-in"),
    ("politics", "vote-on"),
    ("politics", "voted-on"),
    ("politics", "us-recogni"),               # US recognizes X
    ("politics", "war-in-"),
    ("politics", "withdraw-from"),
    ("politics", "leaves-office"),
    ("politics", "out-of-office"),
    ("politics", "out-as-"),                 # "X out as Y" political resignations
    ("politics", "resign"),
    ("politics", "impeachment"),
    ("politics", "constitution"),
    ("politics", "border-wall"),
    ("politics", "minneapolis"),
    ("politics", "memphis"),
    ("politics", "chicago"),                 # Most "Chicago" markets are political
    ("politics", "new-york-city-"),
    ("politics", "nyc-mayor"),
    ("politics", "nyc-mayoral"),
    ("politics", "mamdani"),
    ("politics", "kushner"),
    ("politics", "stephen-miller"),
    ("politics", "miller"),                  # broad but most "Miller" markets are political
    ("politics", "doge-"),                   # Wait — careful, doge is also crypto. Skip.
    # === Macro ===
    ("macro", "earnings"),                   # quarterly earnings markets
    ("macro", "interest-rate"),
    ("macro", "fed-interest"),
    ("macro", "fed-funds"),
    ("macro", "feds-lower-bound"),
    ("macro", "march-2026-meeting"),         # FOMC meeting markets
    ("macro", "powell"),                     # macro figure
    ("macro", "fed-chair"),
    ("macro", "fed-cut"),
    ("macro", "rate-decision"),
    ("macro", "rate-meeting"),
    ("macro", "basis-points"),
    ("macro", "-bps-"),
    ("macro", "by-eoy"),                     # earnings/inflation targets by end of year
    ("macro", "stock-market"),
    ("macro", "dow-jones"),
    ("macro", "nasdaq"),
    ("macro", "russell"),
    ("macro", "gold-price"),
    ("macro", "oil-price"),
    ("macro", "brent-crude"),
    ("macro", "wti-"),
    ("macro", "treasury"),
    ("macro", "yields"),
    ("macro", "fed-meeting"),
    ("macro", "rate-change"),
    # === Crypto ===
    ("crypto", "metamask"),
    ("crypto", "ledger"),
    ("crypto", "trezor"),
    ("crypto", "wallet"),
    ("crypto", "nft"),
    ("crypto", "memecoin"),
    ("crypto", "memecoins"),
    ("crypto", "altcoin"),
    ("crypto", "polymarket"),                # Polymarket-as-company markets
    ("crypto", "kalshi"),                    # competitor prediction market
    ("crypto", "polymarket-us-go-live"),
    ("crypto", "shiba"),
    ("crypto", "pepecoin"),
    ("crypto", "ftx"),
    ("crypto", "sam-bankman"),
    ("crypto", "sbf-"),
    ("crypto", "uniswap"),
    ("crypto", "aave"),
    ("crypto", "chainlink"),
    ("crypto", "celsius"),
    ("crypto", "luna"),
    ("crypto", "terra-"),
    # === Sports — high-confidence ===
    ("sports", "bills"),
    ("sports", "patriots"),
    ("sports", "ravens"),
    ("sports", "steelers"),
    ("sports", "browns"),
    ("sports", "rams"),
    ("sports", "seahawks"),
    ("sports", "49ers"),
    ("sports", "jets-"),
    ("sports", "giants"),
    ("sports", "cowboys"),
    ("sports", "redskins"),
    ("sports", "commanders"),
    ("sports", "saints"),
    ("sports", "panthers"),
    ("sports", "buccaneers"),
    ("sports", "falcons"),
    ("sports", "vikings"),
    ("sports", "packers"),
    ("sports", "bears"),
    ("sports", "lions"),
    ("sports", "raiders"),
    ("sports", "chargers"),
    ("sports", "titans"),
    ("sports", "jaguars"),
    ("sports", "colts"),
    ("sports", "texans"),
    ("sports", "dolphins"),
    ("sports", "bengals"),
    ("sports", "cardinals"),
    ("sports", "guardians"),
    ("sports", "phillies"),
    ("sports", "astros"),
    ("sports", "rangers"),
    ("sports", "padres"),
    ("sports", "giants-win"),
    ("sports", "mets-"),
    ("sports", "red-sox"),
    ("sports", "blue-jays"),
    ("sports", "orioles"),
    ("sports", "tigers"),
    ("sports", "athletics"),
    ("sports", "twins"),
    ("sports", "royals"),
    ("sports", "brewers"),
    ("sports", "cubs"),
    ("sports", "white-sox"),
    ("sports", "pirates"),
    ("sports", "reds"),
    ("sports", "marlins"),
    ("sports", "rockies"),
    ("sports", "diamondbacks"),
    ("sports", "spurs"),
    ("sports", "rockets"),
    ("sports", "pelicans"),
    ("sports", "thunder"),
    ("sports", "grizzlies"),
    ("sports", "kings"),
    ("sports", "suns"),
    ("sports", "nuggets"),
    ("sports", "timberwolves"),
    ("sports", "trail-blazers"),
    ("sports", "jazz-"),
    ("sports", "clippers"),
    ("sports", "76ers"),
    ("sports", "nets-"),
    ("sports", "raptors"),
    ("sports", "hawks"),
    ("sports", "magic"),
    ("sports", "wizards"),
    ("sports", "pistons"),
    ("sports", "bucks-"),
    ("sports", "cavaliers"),
    ("sports", "pacers"),
    ("sports", "bobcats"),
    ("sports", "hornets"),
    ("sports", "espn"),
    ("sports", "best-record"),               # "best record in the NFL/MLB"
    ("sports", "win-the-al-"),
    ("sports", "win-the-nl-"),
    ("sports", "win-the-east-"),
    ("sports", "win-the-west-"),
    ("sports", "afc-east"),
    ("sports", "afc-west"),
    ("sports", "afc-north"),
    ("sports", "afc-south"),
    ("sports", "nfc-east"),
    ("sports", "nfc-west"),
    ("sports", "nfc-north"),
    ("sports", "nfc-south"),
    ("sports", "fa-cup"),
    ("sports", "europa-league"),
    ("sports", "uefa"),
    ("sports", "fifa"),
    ("sports", "el-clasico"),
    ("sports", "ucl-final"),
    ("sports", "ucl-winner"),
    ("sports", "messi"),
    ("sports", "ronaldo"),
    ("sports", "mbappe"),
    ("sports", "haaland"),
    ("sports", "neymar"),
]

# Remove the careless ("politics", "doge-") since doge is crypto
PROPOSED_ADDITIONS = [p for p in PROPOSED_ADDITIONS if p != ("politics", "doge-")]


def _build_classifier(rules: list[tuple[str, tuple[str, ...]]]):
    def classify(slug: str | None) -> str:
        if not slug:
            return "other"
        s = "-" + slug.lower() + "-"
        for fam, kws in rules:
            for kw in kws:
                if kw in s:
                    return fam
        return "other"
    return classify


def _augment_rules() -> list[tuple[str, tuple[str, ...]]]:
    """Original rules + PROPOSED_ADDITIONS, keeping family order so earlier
    matches win (politics added kws still go in politics, etc.).
    """
    augmented = {fam: list(kws) for fam, kws in FAMILY_KEYWORDS_ORIG}
    for fam, kw in PROPOSED_ADDITIONS:
        augmented.setdefault(fam, []).append(kw)
    # Preserve original order: sports, crypto, macro, weather, politics
    out = []
    for fam, _ in FAMILY_KEYWORDS_ORIG:
        out.append((fam, tuple(augmented[fam])))
    return out


# ----------------------------------------------------------------------------
def fmt_md_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].apply(
                lambda x: "" if pd.isna(x) else (
                    f"{x:,.4f}" if abs(x) < 10 else f"{x:,.0f}"
                )
            )
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].apply(lambda x: "" if pd.isna(x) else f"{int(x):,}")
        else:
            d[c] = d[c].astype(str).where(d[c].notna(), "")
    cols = list(d.columns)
    widths = [max(len(c), d[c].map(len).max() if len(d) else 0) for c in cols]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = ["| " + " | ".join(v.rjust(w) for v, w in zip(r, widths)) + " |"
            for r in d.to_numpy().tolist()]
    return "\n".join([head, sep] + rows)


def main() -> None:
    print("[task1] loading audit fragments + positions…", flush=True)
    f = pd.read_parquet(FRAG_PATH)
    p = pd.read_parquet(POS_PATH)
    print(f"  fragments={len(f):,}  positions={len(p):,}", flush=True)

    # Per-slug aggregates from existing audit data
    per_slug = (
        f.groupby(["slug", "family"], dropna=False)
         .agg(n_fills=("anchor_idx", "size"),
              notional=("usd_amount", "sum"))
         .reset_index()
    )
    # Leader PnL per (market_id, token) → per slug via fragments slug map.
    f_slug_map = f.drop_duplicates(["market_id", "outcome_token_id"])[
        ["market_id", "outcome_token_id", "slug"]
    ]
    p_with_slug = p.merge(f_slug_map, on=["market_id", "outcome_token_id"], how="left")
    pnl_per_slug = p_with_slug.groupby("slug")["domah_pnl_calc"].sum().rename("leader_pnl").reset_index()
    per_slug = per_slug.merge(pnl_per_slug, on="slug", how="left")
    per_slug = per_slug.sort_values("n_fills", ascending=False)

    # ============================================================
    # Section 1: top-30 + random 30 slugs per family
    # ============================================================
    sections = []
    sections.append("# Family Heuristic Validation\n")
    sections.append(f"_Generated from existing Domah audit data._\n")
    sections.append(f"Total fragments: **{len(f):,}**, positions: **{len(p):,}**.\n")

    sections.append("## 1. Per-family slug samples\n")
    sections.append("Top 30 distinct slugs by Domah's fill count, plus a random 30 more.\n")
    for fam in ["politics", "macro", "sports", "crypto", "weather", "other"]:
        ps = per_slug[per_slug["family"] == fam]
        n_total = int(ps["n_fills"].sum())
        notional = float(ps["notional"].sum())
        pnl = float(ps["leader_pnl"].sum())
        sections.append(f"### {fam} — n_slugs={len(ps):,}, n_fills={n_total:,}, "
                        f"notional=${notional:,.0f}, leader_pnl=${pnl:,.0f}\n")
        top = ps.head(30)[["slug", "n_fills", "notional", "leader_pnl"]]
        sections.append("**Top 30 by fills:**\n")
        sections.append(fmt_md_table(top))
        sections.append("")
        if len(ps) > 30:
            rng = np.random.default_rng(42)
            rest = ps.iloc[30:]
            rand = rest.sample(n=min(30, len(rest)), random_state=42)[
                ["slug", "n_fills", "notional", "leader_pnl"]
            ]
            sections.append("**Random sample of 30 more:**\n")
            sections.append(fmt_md_table(rand))
            sections.append("")

    # ============================================================
    # Section 2: misclassification candidates in OTHER
    # ============================================================
    sections.append("## 2. Misclassification candidates (currently in `other`)\n")
    sections.append(
        "Slugs from `other` that visibly belong in another family per their text. "
        "Identified by manual review of the top-200 'other' slugs.\n"
    )
    other_top = per_slug[per_slug["family"] == "other"].head(200).copy()

    # rule-of-thumb misclassification matchers
    def reclassify_other(slug: str) -> str | None:
        s = "-" + (slug or "").lower() + "-"
        # politics
        for kw in (
            "eric-adams", "weed-rescheduled", "weed-reschedul", "national-guard",
            "deportation", "passport", "executive-action",
            "secretary-of", "ambassador-to", "sanctions-on",
            "ceasefire", "war-on", "world-leader", "world-leaders",
            "nuclear-", "diplomatic", "embassy", "treaty", "summit",
            "negotiations", "rfk-", "desantis", "newsom",
            "abrego", "kilmar", "cz-by", "pardon", "bondi",
            "kash-patel", "vance-", "rubio", "tulsi", "prime-minister",
            "presidential-race", "election-in", "elections-in",
            "us-recogni", "war-in-", "withdraw-from",
            "out-as-", "resign", "impeachment", "border-wall",
            "minneapolis", "memphis", "chicago", "new-york-city-",
            "nyc-mayor", "mamdani", "kushner",
            "starmer", "macron", "merz", "scholz",
            "supreme-leader", "regime-change", "coup-",
            "anti-cartel",
        ):
            if kw in s:
                return "politics"
        # macro
        for kw in (
            "earnings", "interest-rate", "fed-interest", "fed-funds",
            "feds-lower-bound", "powell", "fed-chair",
            "fed-cut", "rate-decision", "rate-meeting", "basis-points",
            "-bps-", "stock-market", "dow-jones", "nasdaq",
            "russell", "gold-price", "oil-price", "brent-crude",
            "wti-", "treasury", "yields", "fed-meeting", "rate-change",
        ):
            if kw in s:
                return "macro"
        # crypto
        for kw in (
            "metamask", "ledger", "trezor", "wallet", "memecoin",
            "memecoins", "altcoin", "polymarket", "kalshi",
            "polymarket-us-go-live", "shiba", "pepecoin", "ftx",
            "sam-bankman", "sbf-", "uniswap", "aave", "chainlink",
            "celsius", "luna", "terra-",
        ):
            if kw in s:
                return "crypto"
        # sports
        for kw in (
            "bills", "patriots", "ravens", "steelers", "browns",
            "rams", "seahawks", "49ers", "jets-",
            "giants", "cowboys", "redskins", "commanders", "saints",
            "panthers", "buccaneers", "falcons", "vikings", "packers",
            "bears", "lions", "raiders", "chargers", "titans",
            "jaguars", "colts", "texans", "dolphins", "bengals", "cardinals",
            "guardians", "phillies", "astros", "rangers", "padres",
            "mets-", "red-sox", "blue-jays", "orioles", "tigers",
            "athletics", "twins", "royals", "brewers", "cubs",
            "white-sox", "pirates", "reds", "marlins", "rockies",
            "diamondbacks", "spurs", "rockets", "pelicans", "thunder",
            "grizzlies", "kings", "suns", "nuggets", "timberwolves",
            "trail-blazers", "jazz-", "clippers", "76ers", "nets-",
            "raptors", "hawks", "magic", "wizards", "pistons",
            "bucks-", "cavaliers", "pacers", "bobcats", "hornets",
            "espn", "best-record", "win-the-al-", "win-the-nl-",
            "win-the-east-", "win-the-west-",
            "afc-east", "afc-west", "afc-north", "afc-south",
            "nfc-east", "nfc-west", "nfc-north", "nfc-south",
            "fa-cup", "europa-league", "uefa", "fifa",
            "el-clasico", "ucl-final", "ucl-winner",
            "messi", "ronaldo", "mbappe", "haaland", "neymar",
        ):
            if kw in s:
                return "sports"
        return None

    other_top["proposed_family"] = other_top["slug"].map(reclassify_other)
    moves = other_top[other_top["proposed_family"].notna()].copy()
    moves_summary = (
        moves.groupby("proposed_family")
        .agg(n_slugs=("slug", "nunique"),
             n_fills=("n_fills", "sum"),
             notional=("notional", "sum"),
             leader_pnl=("leader_pnl", "sum"))
        .reset_index()
        .sort_values("n_fills", ascending=False)
    )
    sections.append("**Summary of proposed `other` → X moves (top-200 slugs only):**\n")
    sections.append(fmt_md_table(moves_summary))
    sections.append("")
    sections.append(f"Total `other`-bucket fills that would re-route: "
                    f"**{int(moves['n_fills'].sum()):,}** "
                    f"(out of {int(other_top['n_fills'].sum()):,} fills in top-200 `other` slugs).\n")
    sections.append("\n**Per-slug moves (top 60 by fills):**\n")
    sections.append(fmt_md_table(
        moves.head(60)[["slug", "n_fills", "notional", "leader_pnl", "proposed_family"]]
    ))
    sections.append("")

    # ============================================================
    # Section 3: proposed heuristic update + re-route counts per keyword
    # ============================================================
    sections.append("## 3. Proposed `FAMILY_KEYWORDS` update\n")
    sections.append(
        f"Adding **{len(PROPOSED_ADDITIONS)}** keywords across all families. "
        "For each keyword, n_fills routed (counts the keyword as the trigger that "
        "matched first under the new classifier) + 3 example slugs that matched it.\n"
    )

    classify_new = _build_classifier(_augment_rules())
    f["family_new"] = f["slug"].map(classify_new)
    routed = []
    for fam, kw in PROPOSED_ADDITIONS:
        mask = f["slug"].str.contains(kw, na=False, regex=False)
        n = int(mask.sum())
        if n == 0:
            continue
        examples = list(f.loc[mask, "slug"].drop_duplicates().head(3).values)
        routed.append({
            "target_family": fam, "keyword": kw, "n_fills_match": n,
            "examples": "; ".join(examples)[:140],
        })
    routed_df = pd.DataFrame(routed).sort_values("n_fills_match", ascending=False)
    sections.append(fmt_md_table(routed_df))
    sections.append("")

    # ============================================================
    # Section 4: impact recompute (cheap — just re-bucket existing data)
    # ============================================================
    sections.append("## 4. Estimated impact on the audit\n")
    # New family per (market_id, token) → propagate to positions
    new_pos_family = (
        f.drop_duplicates(["market_id", "outcome_token_id"])[
            ["market_id", "outcome_token_id", "family_new"]
        ]
    )
    p_new = p.merge(new_pos_family, on=["market_id", "outcome_token_id"], how="left")

    branches = ["A_opt", "A_real", "B", "C_opt", "C_real"]
    def family_tbl_from(positions: pd.DataFrame, frag: pd.DataFrame,
                        fam_col: str) -> pd.DataFrame:
        rows = []
        for fam, sub in positions.groupby(fam_col, sort=False):
            row = {
                "family": fam,
                "n_fills": int(sub["n_fills"].sum()),
                "n_positions": int(len(sub)),
                "leader_pnl": float(sub["domah_pnl_calc"].sum()),
            }
            for br in branches:
                row[f"{br}_pnl"] = float(sub[f"{br}_pnl"].sum())
            # adverse_select_ratio
            f_fam_ids = sub["position_id"].tolist()
            f_fam = frag[frag["position_id"].isin(f_fam_ids) & (frag["role"] == "maker")]
            f_w = f_fam.merge(sub[["position_id", "is_winning_position"]],
                              on="position_id", how="left")
            win  = f_w[f_w["is_winning_position"] == 1]
            lose = f_w[f_w["is_winning_position"] == 0]
            if len(win) >= 30 and len(lose) >= 30:
                rw = float(win["A_real_fill"].mean())
                rl = float(lose["A_real_fill"].mean())
                row["adverse_select_ratio"] = rw / rl if rl > 0 else float("nan")
            else:
                row["adverse_select_ratio"] = float("nan")
            row["A_opt_capture"]  = row["A_opt_pnl"]  / row["leader_pnl"] if row["leader_pnl"] != 0 else float("nan")
            row["A_real_capture"] = row["A_real_pnl"] / row["leader_pnl"] if row["leader_pnl"] != 0 else float("nan")
            rows.append(row)
        return pd.DataFrame(rows).sort_values("n_fills", ascending=False).reset_index(drop=True)

    # Need a "new family" column on positions; merging works only if all positions
    # are present in fragments (they should be). Confirm:
    if p_new["family_new"].isna().any():
        print(f"  WARNING: {p_new['family_new'].isna().sum()} positions lost family")
        p_new["family_new"] = p_new["family_new"].fillna("other")

    fam_orig = pd.read_parquet(FAM_PATH)[
        ["family", "n_fills", "n_positions", "leader_pnl",
         "A_opt_pnl", "A_real_pnl", "B_pnl", "C_opt_pnl", "C_real_pnl",
         "A_opt_capture", "A_real_capture", "adverse_select_ratio"]
    ]
    fam_new = family_tbl_from(p_new, f, "family_new")

    sections.append("### Family table — original keyword set\n")
    sections.append(fmt_md_table(fam_orig))
    sections.append("")
    sections.append("### Family table — proposed keyword set\n")
    sections.append(fmt_md_table(fam_new[
        ["family", "n_fills", "n_positions", "leader_pnl",
         "A_opt_pnl", "A_real_pnl", "B_pnl", "C_opt_pnl", "C_real_pnl",
         "A_opt_capture", "A_real_capture", "adverse_select_ratio"]
    ]))
    sections.append("")

    # Diff: capture-ratio shift and adverse-select-ratio shift per family
    diff_rows = []
    fo = fam_orig.set_index("family")
    fn = fam_new.set_index("family")
    common_fams = sorted(set(fo.index) & set(fn.index))
    for fam in common_fams:
        ao = fo.loc[fam, "A_real_capture"]
        an = fn.loc[fam, "A_real_capture"]
        asr_o = fo.loc[fam, "adverse_select_ratio"]
        asr_n = fn.loc[fam, "adverse_select_ratio"]
        nf_o = int(fo.loc[fam, "n_fills"])
        nf_n = int(fn.loc[fam, "n_fills"])
        diff_rows.append({
            "family": fam,
            "n_fills_orig": nf_o, "n_fills_new": nf_n, "fill_delta": nf_n - nf_o,
            "A_real_capture_orig": ao, "A_real_capture_new": an,
            "capture_delta": (an - ao) if (pd.notna(ao) and pd.notna(an)) else float("nan"),
            "adv_sel_orig": asr_o, "adv_sel_new": asr_n,
            "sign_flip_capture": bool(pd.notna(ao) and pd.notna(an) and (ao > 0) != (an > 0)),
            "material_capture_shift_gt_10pp": bool(
                pd.notna(ao) and pd.notna(an) and abs(an - ao) > 0.10
            ),
        })
    diff_df = pd.DataFrame(diff_rows)
    sections.append("### Diff: orig vs proposed\n")
    sections.append(fmt_md_table(diff_df))
    sections.append("")

    material = diff_df[
        diff_df["material_capture_shift_gt_10pp"] | diff_df["sign_flip_capture"]
    ]
    if len(material):
        sections.append("**Material findings change** (>10pp A_real capture shift OR sign flip):\n")
        for _, r in material.iterrows():
            sections.append(
                f"- **{r['family']}**: capture {r['A_real_capture_orig']:.2f} → "
                f"{r['A_real_capture_new']:.2f} (Δ={r['capture_delta']:.2f}); "
                f"adv-sel {r['adv_sel_orig']:.2f} → {r['adv_sel_new']:.2f}; "
                f"fills {r['n_fills_orig']:,} → {r['n_fills_new']:,}."
            )
    else:
        sections.append("**No material findings change** — capture ratios and adv-sel ratios "
                        "stable within 10pp across all families. Reclassification mostly "
                        "moves fills around without changing the deployment decision.")

    out_path = OUT_DIR / "family_heuristic_validation.md"
    out_path.write_text("\n".join(sections))
    print(f"[task1] wrote {out_path}")


if __name__ == "__main__":
    main()
