#!/usr/bin/env python
"""Independent token<->identity check against a FRESH Gamma pull. Does not use the five check columns,
audit_market(), or any loader logic — only tokens.parquet as the thing under test, and Gamma as the oracle.

For N seeded-random tokens (stratified by universe), fetch /markets?condition_ids=<cid> and compare:
  asset_id      == clobTokenIds[outcome_index]      <- THE mapping (price series -> token)
  outcome_label == outcomes[outcome_index]           <- side label
  market_slug   == slug ; question == question ; neg_risk == negRisk ; event_slug == events[0].slug
Any asset_id mismatch means a price series is attached to the wrong token/event.

    python scripts/audit_checks/check_mapping_gamma.py [n_per_universe=15] [seed=7]
"""
import sys, json, time, urllib.request, urllib.parse
sys.path.insert(0, ".")
import duckdb, pandas as pd
from epsilon_data._internal import _uri

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
tok = duckdb.connect().execute(f"SELECT * FROM read_parquet('{_uri('tokens.parquet')}') WHERE identity_status='resolved'").df()
for col in ("asset_id", "condition_id", "event_id"):
    tok[col] = tok[col].astype(str)
samp = pd.concat([tok[tok.universe == u].sample(N, random_state=SEED) for u in ("politics_negrisk", "esports")])


def gamma(cid):
    url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"condition_ids": cid})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "epsilon-audit/1"}), timeout=25) as r:
        return json.load(r)


rows = []
for r in samp.itertuples():
    try:
        d = gamma(r.condition_id); time.sleep(0.25)
    except Exception as e:
        rows.append({"universe": r.universe, "condition_id": r.condition_id, "gamma": f"ERROR {e}"}); continue
    if not d:
        rows.append({"universe": r.universe, "condition_id": r.condition_id, "gamma": "NOT FOUND"}); continue
    g = d[0]
    ids = json.loads(g.get("clobTokenIds") or "[]"); outs = json.loads(g.get("outcomes") or "[]")
    i = int(r.outcome_index)
    ev = (g.get("events") or [{}])[0]
    rows.append({
        "universe": r.universe, "condition_id": r.condition_id, "gamma": "ok",
        "asset_id_match": i < len(ids) and ids[i] == r.asset_id,
        "asset_id_in_market_at_all": r.asset_id in ids,
        "outcome_label_match": i < len(outs) and outs[i] == r.outcome_label,
        "slug_match": g.get("slug") == r.market_slug,
        "question_match": (g.get("question") or "").strip() == (r.question or "").strip(),
        "neg_risk_match": bool(g.get("negRisk")) == bool(r.neg_risk),
        "event_slug_match": ev.get("slug") == r.event_slug,
        "event_id_match": str(ev.get("id")) == r.event_id,
        "closed_gamma": g.get("closed"), "closed_ours": bool(r.closed),
        "ours_outcome_label": r.outcome_label, "gamma_outcome": outs[i] if i < len(outs) else None,
    })
df = pd.DataFrame(rows)
ok = df[df.gamma == "ok"]
summary = {"sampled": len(df), "gamma_ok": len(ok), "seed": SEED,
           **{c: {"pass": int(ok[c].sum()), "of": len(ok)} for c in
              ("asset_id_match", "asset_id_in_market_at_all", "outcome_label_match", "slug_match",
               "question_match", "neg_risk_match", "event_slug_match", "event_id_match")},
           "closed_agree": int((ok.closed_gamma == ok.closed_ours).sum())}
print(json.dumps(summary, indent=2))
bad = ok[~(ok.asset_id_match & ok.outcome_label_match & ok.slug_match & ok.event_slug_match)]
print(f"\nrows with ANY mismatch: {len(bad)}")
if len(bad):
    print(bad[["universe", "condition_id", "asset_id_match", "outcome_label_match", "slug_match", "event_slug_match",
               "event_id_match", "ours_outcome_label", "gamma_outcome"]].to_string(index=False, max_colwidth=18))
nf = df[df.gamma != "ok"]
if len(nf):
    print("\nnot resolvable via Gamma:"); print(nf[["universe", "condition_id", "gamma"]].to_string(index=False, max_colwidth=70))
df.to_json(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", orient="records", indent=1)
