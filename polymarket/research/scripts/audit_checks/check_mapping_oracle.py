#!/usr/bin/env python
"""Independent token<->identity check against TWO fresh oracles. No loader logic, no check columns.

  CLOB  https://clob.polymarket.com/markets/{condition_id}   -> tokens[{token_id, outcome, winner}], market_slug,
        question, neg_risk, closed.  Works for closed markets. This is the venue the price feed came from.
  Gamma https://gamma-api.polymarket.com/markets?condition_ids=&closed=true -> events[0].{slug,id}
        (Gamma's condition_ids filter silently drops closed markets unless closed=true is passed.)

Compared per token:
  asset_id  in  CLOB token_ids   AND   CLOB outcome(asset_id) == our outcome_label      <- THE mapping
  market_slug, question, neg_risk, closed (CLOB);   event_slug, event_id (Gamma)

Modes:
  random   N per universe, seeded                     python ... random 30 7
  check3   n_false tokens with check3_negrisk=False + n_true with =True (all politics)
                                                      python ... check3 20 10
"""
import sys, json, time, urllib.request, urllib.parse
sys.path.insert(0, ".")
import duckdb, pandas as pd
from epsilon_data._internal import _uri

MODE = sys.argv[1] if len(sys.argv) > 1 else "random"
A = int(sys.argv[2]) if len(sys.argv) > 2 else 30
Bn = int(sys.argv[3]) if len(sys.argv) > 3 else 7
tok = duckdb.connect().execute(f"SELECT * FROM read_parquet('{_uri('tokens.parquet')}') WHERE identity_status='resolved'").df()
for col in ("asset_id", "condition_id", "event_id"):
    tok[col] = tok[col].astype(str)
if MODE == "random":
    samp = pd.concat([tok[tok.universe == u].sample(A, random_state=Bn) for u in ("politics_negrisk", "esports")])
else:
    samp = pd.concat([tok[tok.check3_negrisk == False].sample(A, random_state=5),
                      tok[tok.check3_negrisk == True].sample(Bn, random_state=5)])


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "epsilon-audit/1"}), timeout=25) as r:
        return json.load(r)


clob_cache, gamma_cache = {}, {}
rows = []
for r in samp.itertuples():
    cid = r.condition_id
    try:
        if cid not in clob_cache:
            clob_cache[cid] = get(f"https://clob.polymarket.com/markets/{cid}"); time.sleep(0.15)
        cm = clob_cache[cid]
    except Exception as e:
        rows.append({"universe": r.universe, "condition_id": cid, "clob": f"ERROR {str(e)[:60]}"}); continue
    if not cm or "tokens" not in cm:
        rows.append({"universe": r.universe, "condition_id": cid, "clob": "NOT FOUND"}); continue
    toks = {t["token_id"]: t for t in cm["tokens"]}
    mine = toks.get(r.asset_id)
    try:
        if cid not in gamma_cache:
            g = get("https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"condition_ids": cid, "closed": "true"}))
            if not g:
                g = get("https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"condition_ids": cid}))
            gamma_cache[cid] = g[0] if g else None; time.sleep(0.15)
        gm = gamma_cache[cid]
    except Exception:
        gm = None
    ev = ((gm or {}).get("events") or [{}])[0]
    rows.append({
        "universe": r.universe, "condition_id": cid, "clob": "ok", "check3": r.check3_negrisk, "check4": r.check4_status,
        "asset_in_clob_market": mine is not None,
        "outcome_label_match": (mine or {}).get("outcome") == r.outcome_label,
        "slug_match": cm.get("market_slug") == r.market_slug,
        "question_match": (cm.get("question") or "").strip() == (r.question or "").strip(),
        "neg_risk_match": bool(cm.get("neg_risk")) == bool(r.neg_risk),
        "closed_match": bool(cm.get("closed")) == bool(r.closed),
        "clob_winner_this_token": (mine or {}).get("winner"),
        "ours_resolved_outcome": r.resolved_outcome, "ours_label": r.outcome_label,
        "gamma_found": gm is not None,
        "event_slug_match": (ev.get("slug") == r.event_slug) if gm else None,
        "event_id_match": (str(ev.get("id")) == r.event_id) if gm else None,
    })
df = pd.DataFrame(rows)
ok = df[df.clob == "ok"].copy()
flds = ["asset_in_clob_market", "outcome_label_match", "slug_match", "question_match", "neg_risk_match", "closed_match"]
summary = {"mode": MODE, "sampled": len(df), "clob_ok": len(ok), "gamma_found": int(ok.gamma_found.sum()),
           **{c: {"pass": int(ok[c].sum()), "of": len(ok)} for c in flds},
           "event_slug_match": {"pass": int(ok.event_slug_match.fillna(False).sum()), "of": int(ok.gamma_found.sum())},
           "event_id_match": {"pass": int(ok.event_id_match.fillna(False).sum()), "of": int(ok.gamma_found.sum())}}
# resolution agreement where CLOB says someone won: our resolved_outcome should equal the CLOB winner's outcome label
won = ok[ok.clob_winner_this_token == True]
summary["clob_winner_rows"] = len(won)
summary["ours_resolved_outcome_equals_clob_winner"] = int((won.ours_resolved_outcome == won.ours_label).sum())
if MODE == "check3":
    for v in (False, True):
        s = ok[ok.check3 == v]
        summary[f"check3={v}"] = {"n": len(s), "asset+outcome ok": int((s.asset_in_clob_market & s.outcome_label_match).sum()),
                                  "slug ok": int(s.slug_match.sum()), "event_slug ok": int(s.event_slug_match.fillna(False).sum())}
print(json.dumps(summary, indent=2))
bad = ok[~(ok.asset_in_clob_market & ok.outcome_label_match & ok.slug_match & ok.question_match)]
print(f"\nrows with a CLOB mismatch: {len(bad)}")
if len(bad):
    print(bad[["universe", "condition_id", "asset_in_clob_market", "outcome_label_match", "slug_match", "question_match", "ours_label"]]
          .to_string(index=False, max_colwidth=18))
nf = df[df.clob != "ok"]
if len(nf): print("\nnot in CLOB:"); print(nf[["universe", "condition_id", "clob"]].to_string(index=False, max_colwidth=70))
