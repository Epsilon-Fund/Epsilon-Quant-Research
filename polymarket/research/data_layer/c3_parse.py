"""C3 · Parse. Reads ONLY gamma_cache/ (never the network). Produces tokens_raw.parquet.

One row per token (asset_id). Identity from Gamma where resolved; null + identity_status
='unresolved' for .notfound conditions (never dropped). Length-2/alignment failures go to
quarantine.csv, not the table. Every id is handled as text — a 77-digit id must never touch
a numeric type.
"""
import os, json, csv
import duckdb, pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma")
CACHE = os.path.join(DATA, "gamma_cache")
DQ = DATA.replace(chr(92), "/")

con = duckdb.connect()
# universe + expected asset_ids per condition, from targets (all as text)
uni = {r[0]: r[1] for r in con.execute(
    f"SELECT condition_id, universe FROM read_parquet('{DQ}/targets_rollup.parquet')").fetchall()}
tgt_assets = {}
for cid, aid in con.execute(
    f"SELECT condition_id, asset_id FROM read_parquet('{DQ}/targets.parquet')").fetchall():
    tgt_assets.setdefault(cid, []).append(aid)

def jdec(s):
    if isinstance(s, str):
        try: return json.loads(s)
        except json.JSONDecodeError: return None
    return s

rows = []
quar = []
n_json = n_nf = 0
for fn in os.listdir(CACHE):
    path = os.path.join(CACHE, fn)
    if fn.endswith(".notfound"):
        n_nf += 1
        cid = fn[:-9]
        for i, aid in enumerate(sorted(tgt_assets.get(cid, []))):
            rows.append(dict(asset_id=str(aid), condition_id=cid, universe=uni.get(cid),
                             identity_status="unresolved", outcome_index=i))
        continue
    if not fn.endswith(".json"):
        continue
    n_json += 1
    cid = fn[:-5]
    m = json.loads(open(path, encoding="utf-8").read())
    outcomes = jdec(m.get("outcomes")); ctok = jdec(m.get("clobTokenIds")); oprices = jdec(m.get("outcomePrices")) or []
    # assertion: length 2 and aligned
    if not (isinstance(outcomes, list) and isinstance(ctok, list) and len(outcomes) == 2 and len(ctok) == 2):
        quar.append(dict(condition_id=cid, reason=f"outcomes/clobTokenIds not length-2 (o={outcomes}, t_len={len(ctok) if isinstance(ctok,list) else 'NA'})"))
        continue
    ev = (m.get("events") or [{}])[0]
    closed = m.get("closed"); uma = m.get("umaResolutionStatus")
    for i in range(2):
        price = oprices[i] if i < len(oprices) else None
        label = str(outcomes[i])
        rows.append(dict(
            asset_id=str(ctok[i]),                 # TEXT — never numeric
            condition_id=cid, universe=uni.get(cid),
            gamma_market_id=str(m.get("id")),
            question=m.get("question"), slug=m.get("slug"),
            neg_risk=m.get("negRisk"),
            market_end_date=m.get("endDate"),
            event_id=str(ev.get("id")) if ev.get("id") is not None else None,
            event_title=ev.get("title"), event_slug=ev.get("slug"),
            event_end_date=ev.get("endDate"), event_neg_risk=ev.get("negRisk"),
            outcome_label=label, outcome_index=i,
            outcome=("YES" if label.lower() == "yes" else "NO" if label.lower() == "no" else None),
            outcome_price=price,
            is_winner=(str(price) == "1") if (closed and price is not None) else None,
            closed=closed, uma_status=uma, closed_time=m.get("closedTime"),
            identity_status="resolved",
        ))

df = pd.DataFrame(rows)
# stable column order
cols = ["asset_id","condition_id","universe","identity_status","outcome_index","outcome","outcome_label",
        "question","slug","neg_risk","gamma_market_id","market_end_date",
        "event_id","event_title","event_slug","event_end_date","event_neg_risk",
        "outcome_price","is_winner","closed","uma_status","closed_time"]
df = df.reindex(columns=cols)
df.to_parquet(os.path.join(DATA, "tokens_raw.parquet"))
if quar:
    with open(os.path.join(DATA, "quarantine.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition_id","reason"]); w.writeheader(); w.writerows(quar)

print(f"cache: {n_json} json, {n_nf} notfound")
print(f"tokens_raw.parquet: {len(df)} rows | resolved={ (df.identity_status=='resolved').sum() } "
      f"unresolved={ (df.identity_status=='unresolved').sum() } | quarantined_conditions={len(quar)}")
print("outcome distribution:", df.outcome.value_counts(dropna=False).to_dict())
print("per-universe rows:", df.universe.value_counts(dropna=False).to_dict())
print("OK C3")
