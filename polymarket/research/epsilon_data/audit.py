"""audit_market(ref) — one implementation behind three surfaces (dashboard button, CLI command,
direct call). It produces a report + evidence + a recommendation. It is HUMAN-DECIDED: it NEVER
writes to exclusions.csv. Only write_exclusion() writes, and only on an explicit call/click.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import _internal as _i
from .tape import load_l1, load_trades, load_pair

# The one real multi-hour capture outage (both universes), measured from the parquet rather than
# from coverage.parquet's hour granularity: the last L1 event before the gap is 2026-06-22
# 14:03:30Z and the first after is 2026-06-23 08:29:10Z (both universes agree to within 0.5 s and
# 0.01 s respectively). 18h26m, not the ~17h from 15:00 the docs used to say — there are zero
# rows between 14:03:31 and 15:00.
_OUTAGE = (pd.Timestamp("2026-06-22T14:03:30Z"), pd.Timestamp("2026-06-23T08:29:10Z"))
# How far outside the outage a long gap may extend and still count as explained by it. A token's
# gap runs from its last quote before the outage to its first after, so some slack is needed for a
# sparse tape — but the gap must be CONTAINED in the widened window, not merely overlap it,
# or a months-long hole that happens to span 06-22 would be silently excused.
_OUTAGE_SLACK = pd.Timedelta("1h")
_EXCL_HEADER = "asset_id,scope,reason,date,who"


@dataclass
class Check:
    name: str
    level: str          # "ok" | "note" | "bad"
    detail: str


@dataclass
class AuditResult:
    ref: str
    asset_id: str
    condition_id: str
    question: str
    universe: str
    verdict: str        # "looks fine" | "worth a look" | "recommend excluding"
    reason: str
    checks: list = field(default_factory=list)
    recommend_scope: str = "none"     # "none" | "token" | "market"
    recommend_reason: str = ""
    exclusion_lines: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["checks"] = [asdict(c) if isinstance(c, Check) else c for c in self.checks]
        return d


def _excl_line(asset_id, scope, reason, who="operator"):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reason = str(reason).replace(",", ";")
    return f"{asset_id},{scope},{reason},{date},{who}"


def audit_market(ref) -> AuditResult:
    """Run every cheap check that could indicate a token's series is wrong; return a verdict,
    the evidence behind it, a recommended scope with the trade-off stated, and the exact
    exclusions.csv line(s) to use. Never writes anything."""
    t = _i.tokens()
    ref = str(ref)
    if (t["condition_id"] == ref).any():   # accept a condition_id (a market) -> its side-A token
        aid = t[t["condition_id"] == ref].sort_values("outcome_index")["asset_id"].iloc[0]
    else:
        aid = _i.resolve_ref(ref)
    row = _i.token_row(aid)
    cid = row["condition_id"]
    mkt = t[t["condition_id"] == cid].sort_values("outcome_index")
    checks: list[Check] = []

    # ---- identity ----
    idbad = []
    for k in ("check1_roundtrip", "check2_pairing", "check5_indep"):
        v = row.get(k)
        if v is False:
            idbad.append(k)
    comp = row.get("complement_asset_id")
    comp_present = bool(comp) and (t["asset_id"] == str(comp)).any()
    checks.append(Check("identity", "bad" if idbad else "ok",
                        f"identity_status={row.get('identity_status')}; failed={idbad or 'none'}; "
                        f"complement {'present' if comp_present else 'MISSING'}; check4={row.get('check4_status')}"))

    # ---- load tapes ----
    l1 = load_l1(aid); tr = load_trades(aid)
    if l1.empty:
        checks.append(Check("data", "note", "no L1 events (quiet/unobserved token)"))
        res = AuditResult(str(ref), aid, cid, row.get("question"), row["universe"],
                          "worth a look", "no L1 data to judge the series",
                          checks, "none", "cannot recommend on an empty series", [])
        return res

    # ---- value sanity ----
    mid = l1["mid"].to_numpy(); bb = l1["best_bid"].to_numpy(); ba = l1["best_ask"].to_numpy()
    n = len(mid)
    out01 = int(((mid <= 0) | (mid >= 1)).sum())
    crossed = int((bb > ba + 1e-9).sum())
    sp0 = int((l1["spread_c"] <= 0).sum()); sp100 = int((l1["spread_c"] >= 100).sum())
    moves = int(pd.Series(mid).round(6).nunique())
    jump = float(np.abs(np.diff(mid)).max()) if n > 1 else 0.0
    # severity is FRACTION-based: a handful of crossed/out-of-range ticks (intra-ms ordering, a
    # stale L1) is a NOTE, not a reason to condemn a market with millions of rows.
    def frac(x): return x / max(n, 1)
    lvl = "ok"; sane_bits = []
    lvls = {"ok": 0, "note": 1, "bad": 2}
    def bump(l):
        nonlocal lvl
        if lvls[l] > lvls[lvl]: lvl = l
    if crossed:
        sane_bits.append(f"{crossed} crossed bid>ask ({frac(crossed):.2%})"); bump("bad" if frac(crossed) > 0.005 else "note")
    if out01:
        sane_bits.append(f"{out01} mid outside (0,1) ({frac(out01):.2%})"); bump("bad" if frac(out01) > 0.005 else "note")
    if moves <= 1:
        sane_bits.append("mid never moves"); bump("note")
    if jump > 0.5:
        sane_bits.append(f"max 1-step mid jump {jump*100:.0f}¢"); bump("note")
    # spread=0 (locked/one-tick) and =100¢ (one-sided) are common and NOT defects — report only.
    checks.append(Check("value sanity", lvl,
                        (", ".join(sane_bits) if sane_bits else "no crossed/out-of-range/frozen values") +
                        f"; spread=0 on {frac(sp0):.0%}, =100¢ on {frac(sp100):.0%} (informational)"))

    # ---- internal consistency (pair sum ~1) ----
    try:
        pair = load_pair(cid)
        if not pair.empty and set(pair.columns) >= {0, 1}:
            both = pair.dropna()
            dev = (both[0] + both[1] - 1).abs()
            worst = float(dev.max()); med = float(dev.median())
            lvl = "ok" if med < 0.02 else "note" if med < 0.05 else "bad"
            checks.append(Check("pair sum≈1", lvl, f"median |A+B-1|={med:.3f}, worst={worst:.3f} over {len(both):,} pts"))
        else:
            checks.append(Check("pair sum≈1", "note", "complement has no aligned data"))
    except Exception as e:
        checks.append(Check("pair sum≈1", "note", f"pair unavailable: {str(e)[:60]}"))

    # ---- continuity ----
    # A quiet market naturally has a sparse tape, so a >1h gap is NOT a defect. Only a LONG gap
    # (>12h) that is not the known outage is worth a look.
    ts_sorted = l1["ts"].sort_values()
    dt = ts_sorted.diff()
    big = int((dt > pd.Timedelta("1h")).sum())
    unexplained_long = 0; max_gap_h = float(dt.max().total_seconds() / 3600) if dt.notna().any() else 0.0
    for ts_end, g in zip(ts_sorted.iloc[1:], dt.iloc[1:]):
        if g > pd.Timedelta("12h"):
            ts_start = ts_end - g
            explained = (ts_start >= _OUTAGE[0] - _OUTAGE_SLACK) and (ts_end <= _OUTAGE[1] + _OUTAGE_SLACK)
            if not explained:
                unexplained_long += 1
    checks.append(Check("continuity", "note" if unexplained_long else "ok",
                        f"{big} gaps >1h (normal for a quiet market); {unexplained_long} unexplained gaps >12h; "
                        f"max gap {max_gap_h:.1f}h"))

    # ---- trades vs quotes ----
    tr_note = "no trades"
    tvl = "ok"
    if not tr.empty:
        m = pd.merge_asof(tr.sort_values("timestamp_ms")[["timestamp_ms", "price", "size", "side"]],
                          l1.sort_values("timestamp_ms")[["timestamp_ms", "best_bid", "best_ask"]],
                          on="timestamp_ms", direction="backward")
        no_quote = int(m["best_bid"].isna().sum())
        mm = m.dropna(subset=["best_bid"])
        outside = int(((mm.price < mm.best_bid - 0.01) | (mm.price > mm.best_ask + 0.01)).sum())
        frac = outside / max(len(mm), 1)
        tvl = "ok" if frac < 0.02 else "note" if frac < 0.10 else "bad"
        tr_note = f"{outside}/{len(mm)} trades outside touch (>1¢), {no_quote} with no prior quote"
    checks.append(Check("trades vs quotes", tvl, tr_note))

    # ---- volume shape ----
    vs = "ok"; vbits = []
    if not tr.empty:
        tot = float(tr["size"].sum()) or 1.0
        top = float(tr["size"].max()) / tot
        if top > 0.5: vs = "note"; vbits.append(f"one print = {top:.0%} of volume")
        if "transaction_hash" in tr.columns:
            dups = int(tr["transaction_hash"].duplicated().sum())
            if dups: vbits.append(f"{dups} duplicate tx hashes")
    checks.append(Check("volume shape", vs, ", ".join(vbits) if vbits else "no dominant print / dup hashes"))

    # ---- resolution ----
    rlvl = "ok"; rdet = "not resolved / not applicable"
    if row.get("check4_status") == "inverted":
        rlvl = "bad"; rdet = "INVERTED: the Gamma-winning token did not converge to 1 (one of the 14)"
    elif row.get("check4_status") == "converged_correct":
        rdet = "winner converged to ~1, loser to ~0"
    elif row.get("check4_status") in ("near_half", "no_convergence"):
        rlvl = "note"; rdet = f"resolved but {row.get('check4_status')} (book didn't reach the extreme)"
    checks.append(Check("resolution", rlvl, rdet))

    # ---- verdict ----
    levels = [c.level for c in checks]
    if "bad" in levels:
        verdict = "recommend excluding"
    elif "note" in levels:
        verdict = "worth a look"
    else:
        verdict = "looks fine"
    firstbad = next((c for c in checks if c.level == "bad"), None)
    firstnote = next((c for c in checks if c.level == "note"), None)
    reason = (firstbad or firstnote).detail if (firstbad or firstnote) else "all checks passed"

    # ---- recommendation + exclusion lines ----
    if verdict == "recommend excluding":
        scope = "market"
        rec_reason = ("Exclude the whole MARKET (both tokens): excluding a single token orphans its "
                      "complement and breaks pair views and NegRisk sums; excluding the market keeps the "
                      "dataset internally consistent.")
        lines = [_excl_line(a, "market", reason, "operator") for a in mkt["asset_id"]]
    elif verdict == "worth a look":
        scope = "none"
        rec_reason = "A human should eyeball it before deciding; no automatic recommendation."
        lines = [_excl_line(aid, "token", reason, "operator") + "   # only if you decide token-scope",
                 _excl_line(mkt["asset_id"].iloc[0], "market", reason, "operator") + "   # market-scope (also add the complement)"]
    else:
        scope = "none"; rec_reason = "Series looks sound; keep it."; lines = []

    return AuditResult(str(ref), aid, cid, row.get("question"), row["universe"],
                       verdict, reason, checks, scope, rec_reason, lines)


def write_exclusion(asset_id, scope="market", reason="", who="operator", root=None):
    """Append an exclusion to exclusions.csv. EXPLICIT ACTION ONLY — never called by audit_market.
    Append-only; reversible by deleting the line. Only meaningful for a LOCAL data root (the file is
    hand-edited); refuses an s3 root. Returns the line written."""
    from .config import data_root
    base = Path(root or data_root())
    if str(root or data_root()).startswith("s3://"):
        raise RuntimeError("exclusions.csv is edited on the LOCAL copy; point at a local data root.")
    path = base / "exclusions.csv"
    line = _excl_line(str(asset_id), scope, reason, who)
    header_needed = (not path.exists()) or (_EXCL_HEADER not in path.read_text(encoding="utf-8"))
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# exclusions.csv — hand-edited operator instrument, applied at LOAD time only.\n")
            f.write("# Append one row to hide an asset_id; delete the row to restore it. Nothing auto-excludes.\n")
            f.write(_EXCL_HEADER + "\n")
        f.write(line + "\n")
    _i._exclusions_cached.cache_clear()
    return line
