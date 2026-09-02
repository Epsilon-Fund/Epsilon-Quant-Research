#!/usr/bin/env python
"""Does the price series attached to a token behave like that token's KNOWN resolution?

Oracle = CLOB /markets/{condition_id}: tokens[].winner (independent of our resolved_outcome column).
Tape  = direct DuckDB read of library l1 (not the loader): median of the LAST 20 touch-moves for the
CLOB-winner token and the CLOB-loser token. Winner should end near 1, loser near 0.
A MAPPING error would show as the CLOB winner's series ending near 0 (series attached to the wrong token).

Two samples: 10 markets with check4_status='converged_correct' (tape carries the signal) and 10 random
resolved markets (descriptive: 70% of the dataset is near_half by construction, so non-convergence there is
the documented stale-book phenomenon, not a mapping failure).

    python scripts/audit_checks/check_resolution_convergence.py [seed=11]
"""
import sys, json, time, urllib.request
sys.path.insert(0, ".")
import duckdb, pandas as pd, numpy as np
from epsilon_data._internal import _uri

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 11
c = duckdb.connect()
tok = c.execute(f"SELECT * FROM read_parquet('{_uri('tokens.parquet')}') WHERE closed AND identity_status='resolved'").df()
for col in ("asset_id", "condition_id"): tok[col] = tok[col].astype(str)
two = tok.groupby("condition_id").asset_id.count(); two = set(two[two == 2].index)
tok = tok[tok.condition_id.isin(two) & (tok.n_l1_events >= 50)]
conv = tok[tok.check4_status == "converged_correct"].condition_id.drop_duplicates().sample(10, random_state=SEED).tolist()
rnd = tok.condition_id.drop_duplicates().sample(10, random_state=SEED + 1).tolist()


def clob(cid):
    with urllib.request.urlopen(urllib.request.Request(f"https://clob.polymarket.com/markets/{cid}",
                                headers={"User-Agent": "epsilon-audit/1"}), timeout=25) as r:
        return json.load(r)


def final_mid(asset_id, universe, k=20):
    g = _uri("l1", f"universe={universe}", "*", "*.parquet")
    s = c.execute(f"SELECT mid FROM read_parquet('{g}') WHERE CAST(asset_id AS VARCHAR)='{asset_id}' "
                  f"ORDER BY timestamp_ms DESC, received_ns DESC LIMIT {k}").df().mid
    return float(s.median()) if len(s) else np.nan


rows = []
for label, cids in (("converged_correct", conv), ("random_resolved", rnd)):
    for cid in cids:
        m = tok[tok.condition_id == cid].sort_values("outcome_index"); uni = m.universe.iloc[0]
        try:
            d = clob(cid); time.sleep(0.15)
        except Exception as e:
            rows.append({"sample": label, "condition_id": cid, "status": f"ERROR {str(e)[:50]}"}); continue
        toks = d.get("tokens") or []
        win = [t for t in toks if t.get("winner")]
        if len(toks) != 2 or len(win) != 1:
            rows.append({"sample": label, "condition_id": cid, "status": f"no single winner in CLOB ({len(win)})"}); continue
        w = win[0]; l = [t for t in toks if t is not w][0]
        rows.append({"sample": label, "condition_id": cid, "status": "ok", "universe": uni, "check4": m.check4_status.iloc[0],
                     "clob_winner_outcome": w["outcome"], "ours_resolved_outcome": m.resolved_outcome.iloc[0],
                     "resolved_outcome_agrees": m.resolved_outcome.iloc[0] == w["outcome"],
                     "winner_token_in_ours": w["token_id"] in set(m.asset_id),
                     "final_mid_winner": final_mid(w["token_id"], uni), "final_mid_loser": final_mid(l["token_id"], uni),
                     "n_l1_winner": int(m[m.asset_id == w["token_id"]].n_l1_events.iloc[0]) if w["token_id"] in set(m.asset_id) else 0})
df = pd.DataFrame(rows)
ok = df[df.status == "ok"].copy()
if ok.empty:
    print("no usable markets"); print(df.to_string()); sys.exit(1)
ok["inverted_vs_clob"] = (ok.final_mid_winner < 0.1) & (ok.final_mid_loser > 0.9)
for label in ("converged_correct", "random_resolved"):
    s = ok[ok["sample"] == label]
    print(f"\n=== {label}: {len(s)} markets ===")
    print(json.dumps({"resolved_outcome_agrees_with_clob": int(s.resolved_outcome_agrees.sum()),
                      "winner_token_present_in_ours": int(s.winner_token_in_ours.sum()),
                      "winner_final_mid>0.9": int((s.final_mid_winner > 0.9).sum()),
                      "loser_final_mid<0.1": int((s.final_mid_loser < 0.1).sum()),
                      "INVERTED_vs_clob (mapping-error signature)": int(s.inverted_vs_clob.sum()),
                      "median_final_mid_winner": round(float(s.final_mid_winner.median()), 3)}, indent=1))
    print(s[["universe", "clob_winner_outcome", "ours_resolved_outcome", "check4", "final_mid_winner", "final_mid_loser", "n_l1_winner"]]
          .to_string(index=False))
nf = df[df.status != "ok"]
if len(nf): print("\nskipped:"); print(nf[["sample", "condition_id", "status"]].to_string(index=False, max_colwidth=60))
