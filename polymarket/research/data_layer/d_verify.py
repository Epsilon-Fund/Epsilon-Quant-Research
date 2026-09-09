"""D · Verify the mapping — THE GATE. Separate program from C3.

Assume the mapping is wrong until the price data proves otherwise. Five checks, each
recorded per token in tokens_raw. Prices come from the archive `bba` table (best_bid/
best_ask) via DuckDB httpfs — one aggregate pass gives per-asset median/first/last mid.
Builds no data table. Writes report inputs + one figure; augments tokens_raw with check cols.
"""
import os, sys, json, time, configparser
import duckdb, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma"); DQ = DATA.replace(chr(92), "/")
REP = r"C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\stepD"
os.makedirs(REP, exist_ok=True)
LIVE = r"C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\stepA\live_universe.json"
TOL_PAIR = 0.04     # |mid_A+mid_B - 1| tolerance: ~2x a typical 1-2c spread + staleness slack
TOL_NEG = 0.05      # |sum(YES mids over event) - 1|
WIN_HI, LOSE_LO = 0.90, 0.10

cfg = configparser.ConfigParser(); cfg.read(r"C:\Users\alvar\AppData\Roaming\rclone\rclone.conf")
r = cfg["r2"]; ep = r["endpoint"].replace("https://", "")
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"""CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{r['access_key_id']}',
  SECRET '{r['secret_access_key']}', ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);""")

# ---- price stats per asset from bba (cached) ----
PS = os.path.join(DATA, "price_stats.parquet")
if not os.path.exists(PS):
    from datetime import date, timedelta
    base = "s3://epsilon-polymarket-data/parquet"; d0, d1 = date(2026, 6, 19), date(2026, 8, 21)
    globs = []
    for n in range((d1 - d0).days + 1):
        d = (d0 + timedelta(n)).isoformat()
        globs += [f"{base}/{d}/politics_negrisk/bba_politics_negrisk_*.parquet",
                  f"{base}/{d}/esports/bba_esports_*.parquet"]
    print(f"price scan over bba ({len(globs)} globs)…", flush=True); t0 = time.time()
    con.execute(f"""CREATE TEMP TABLE ps AS
      WITH v AS (SELECT asset_id, timestamp_ms, (best_bid+best_ask)/2.0 mid
                 FROM read_parquet({globs!r}, union_by_name=true)
                 WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL
                   AND best_ask>=best_bid AND best_bid>=0 AND best_ask<=1)
      SELECT asset_id, COUNT(*) n_obs, median(mid) median_mid,
             arg_min(mid,timestamp_ms) first_mid, arg_max(mid,timestamp_ms) last_mid,
             max(timestamp_ms) last_ts FROM v GROUP BY asset_id""")
    con.execute(f"COPY ps TO '{DQ}/price_stats.parquet' (FORMAT parquet)")
    print(f"price scan done in {time.time()-t0:.0f}s", flush=True)
ps = con.execute(f"SELECT * FROM read_parquet('{DQ}/price_stats.parquet')").df()
ps["asset_id"] = ps["asset_id"].astype(str)
pmap = ps.set_index("asset_id")

tok = pd.read_parquet(os.path.join(DATA, "tokens_raw.parquet"))
tok["asset_id"] = tok["asset_id"].astype(str)
tgt = pd.read_parquet(os.path.join(DATA, "targets.parquet"))
tgt["asset_id"] = tgt["asset_id"].astype(str); tgt["condition_id"] = tgt["condition_id"].astype(str)
tgt_by_cid = tgt.groupby("condition_id")["asset_id"].apply(lambda s: set(s)).to_dict()

def med(aid):
    return pmap.loc[aid, "median_mid"] if aid in pmap.index else np.nan
def last(aid):
    return pmap.loc[aid, "last_mid"] if aid in pmap.index else np.nan
tok["median_mid"] = tok["asset_id"].map(med)
tok["last_mid"]   = tok["asset_id"].map(last)

# ---- Check 1: round-trip ids (Gamma vs archive), per resolved condition ----
res = tok[tok.identity_status == "resolved"]
c1 = {}
for cid, g in res.groupby("condition_id"):
    c1[cid] = (set(g.asset_id) == tgt_by_cid.get(cid, set()))
tok["check1_roundtrip"] = tok["condition_id"].map(c1)

# ---- Check 2: pairing mid_A+mid_B ≈ 1 ----
c2 = {}
for cid, g in res.groupby("condition_id"):
    if g.median_mid.notna().all() and len(g) == 2:
        c2[cid] = abs(g.median_mid.sum() - 1.0) <= TOL_PAIR
tok["check2_pairing"] = tok["condition_id"].map(c2)

# ---- Check 3: NegRisk politics — YES mids over an event sum ≈ 1 ----
pol = res[(res.universe == "politics_negrisk") & (res.outcome == "YES")]
ev_sum = pol.groupby("event_id")["median_mid"].agg(["sum", "count", lambda s: s.notna().sum()])
ev_sum.columns = ["yes_sum", "n_mkts", "n_priced"]
ev_pass = {ev: (abs(row.yes_sum - 1.0) <= TOL_NEG) for ev, row in ev_sum.iterrows() if row.n_priced == row.n_mkts and row.n_mkts >= 2}
# map back to tokens via event
tok["check3_negrisk"] = tok["event_id"].map(ev_pass)

# ---- Check 4: resolution — winner→1, loser→0 ----
resolved = res[(res.closed == True) & (res.is_winner.notna())]
c4 = {}
for cid, g in resolved.groupby("condition_id"):
    if len(g) == 2 and g.last_mid.notna().all():
        w = g[g.is_winner == True]; l = g[g.is_winner == False]
        if len(w) == 1 and len(l) == 1:
            c4[cid] = (w.last_mid.iloc[0] >= WIN_HI and l.last_mid.iloc[0] <= LOSE_LO)
tok["check4_resolution"] = tok["condition_id"].map(c4)

# ---- Check 5: independent source (live_universe.json, ~140) ----
lu = json.load(open(LIVE, encoding="utf-8"))
lu_by_cid = {m["condition_id"]: m for m in lu["markets"]}
c5 = {}
for cid, m in lu_by_cid.items():
    g = res[res.condition_id == cid]
    if g.empty:
        c5[cid] = None; continue
    q_ok = (g.question.iloc[0] == m.get("question"))
    s_ok = (g.slug.iloc[0] == m.get("slug"))
    n_ok = (bool(g.neg_risk.iloc[0]) == bool(m.get("neg_risk")))
    pair_ok = (set(g.asset_id) == set(str(a) for a in m.get("asset_ids", [])))
    c5[cid] = bool(q_ok and s_ok and n_ok and pair_ok)
tok["check5_indep"] = tok["condition_id"].map(c5)

tok.to_parquet(os.path.join(DATA, "tokens_raw.parquet"))  # augmented with check cols

# ---------- REPORT NUMBERS ----------
def rate(d, uni_cids=None):
    vals = [v for k, v in d.items() if v is not None and (uni_cids is None or k in uni_cids)]
    return sum(vals), len(vals)
cid_uni = res.drop_duplicates("condition_id").set_index("condition_id")["universe"].to_dict()
pol_cids = {c for c, u in cid_uni.items() if u == "politics_negrisk"}
esp_cids = {c for c, u in cid_uni.items() if u == "esports"}

out = []
out.append("=== COVERAGE ===")
nres = tok[tok.identity_status == "resolved"].condition_id.nunique()
nunr = tok[tok.identity_status == "unresolved"].condition_id.nunique()
out.append(f"targets conditions: {tok.condition_id.nunique()} | gamma-resolved: {nres} | unresolved(.notfound): {nunr}")
unr = tok[tok.identity_status == "unresolved"].drop_duplicates("condition_id")
out.append(f"unresolved by universe: {unr.universe.value_counts().to_dict()}")

for name, d in [("check1_roundtrip", c1), ("check2_pairing", c2), ("check4_resolution", c4)]:
    p_pass, p_n = rate(d, pol_cids); e_pass, e_n = rate(d, esp_cids)
    out.append(f"{name}: politics {p_pass}/{p_n}  esports {e_pass}/{e_n}  (conditions judged)")
p3, n3 = rate(ev_pass)
out.append(f"check3_negrisk (politics events): {p3}/{n3} events pass (|YES-sum-1|<= {TOL_NEG})")
p5, n5 = rate(c5)
out.append(f"check5_indep (live_universe): {p5}/{n5} conditions agree")
# check-4 coverage: fraction of resolved tokens with last_mid
res_tok = tok[(tok.closed == True) & (tok.is_winner.notna())]
for u in ("politics_negrisk", "esports"):
    g = res_tok[res_tok.universe == u]
    cov = g.last_mid.notna().mean() if len(g) else float("nan")
    out.append(f"check4 coverage {u}: {g.last_mid.notna().sum()}/{len(g)} resolved tokens have a mid ({cov:.1%})")

print("\n".join(out))
open(os.path.join(REP, "verify_numbers.txt"), "w", encoding="utf-8").write("\n".join(out))

# ---------- TEN MAPPED MARKETS ----------
def sample_markets(cids, k):
    picks = []
    for cid in cids:
        g = res[res.condition_id == cid].sort_values("outcome_index")
        if len(g) == 2 and g.median_mid.notna().all():
            picks.append(g)
        if len(picks) >= k: break
    return picks
ex = []
ex += sample_markets(list(pol_cids), 5)
ex += sample_markets([c for c in esp_cids if c in c4 and c4[c]], 5)
lines = []
for g in ex:
    r0 = g.iloc[0]
    lines.append(f"\n[{r0.universe}] {r0.question}")
    lines.append(f"  event: {r0.event_title}  slug: {r0.slug}")
    for _, t in g.iterrows():
        lines.append(f"    {t.outcome_label:28s} asset={t.asset_id[:20]}… median_mid={t.median_mid:.3f} last_mid={t.last_mid:.3f} winner={t.is_winner}")
open(os.path.join(REP, "ten_markets.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote ten_markets.txt")

# ---------- FIGURE: resolved markets, both sides mid to resolution ----------
# pick up to 3 esports + up to 3 politics resolved markets that passed check4
def pick_resolved(cids, k):
    o = []
    for c in cids:
        if c4.get(c):
            o.append(c)
        if len(o) >= k: break
    return o
fig_cids = pick_resolved(list(esp_cids), 3) + pick_resolved(list(pol_cids), 3)
# one batched pull for ALL figure assets (avoid a full-window scan per token)
fig_assets = list(res[res.condition_id.isin(fig_cids)].asset_id.unique())
ts_by_asset = {}
if fig_assets:
    from datetime import date, timedelta
    base = "s3://epsilon-polymarket-data/parquet"; d0, d1 = date(2026, 6, 19), date(2026, 8, 21)
    globs = []
    for nn in range((d1 - d0).days + 1):
        d = (d0 + timedelta(nn)).isoformat()
        globs += [f"{base}/{d}/politics_negrisk/bba_politics_negrisk_*.parquet",
                  f"{base}/{d}/esports/bba_esports_*.parquet"]
    idlist = "','".join(fig_assets)
    dfa = con.execute(f"""SELECT asset_id, timestamp_ms, (best_bid+best_ask)/2.0 mid
        FROM read_parquet({globs!r}, union_by_name=true)
        WHERE asset_id IN ('{idlist}') AND best_bid IS NOT NULL AND best_ask IS NOT NULL
        ORDER BY asset_id, timestamp_ms""").df()
    dfa["asset_id"] = dfa["asset_id"].astype(str)
    for aid, gg in dfa.groupby("asset_id"): ts_by_asset[aid] = gg
if fig_cids:
    n = len(fig_cids); fig, axes = plt.subplots(n, 1, figsize=(11, 2.6*n), squeeze=False)
    for ax, cid in zip(axes[:, 0], fig_cids):
        g = res[res.condition_id == cid].sort_values("outcome_index")
        for _, t in g.iterrows():
            d = ts_by_asset.get(t.asset_id)
            if d is None or d.empty: continue
            col = "tab:green" if t.is_winner else "tab:red"
            ax.plot(pd.to_datetime(d.timestamp_ms, unit="ms"), d.mid*100, color=col, lw=0.8,
                    label=f"{t.outcome_label[:18]} ({'WON' if t.is_winner else 'lost'})")
        ax.set_ylim(-5, 105); ax.axhline(100, color="grey", ls=":", lw=0.6); ax.axhline(0, color="grey", ls=":", lw=0.6)
        ax.set_ylabel("mid (¢)"); ax.set_title(f"{g.iloc[0].universe}: {g.iloc[0].question[:70]}", fontsize=9)
        ax.legend(fontsize=7, loc="center left"); ax.grid(True, alpha=0.2)
    fig.suptitle("Step D — resolved markets: winner should walk to 100¢, loser to 0¢", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(REP, "resolution_orientation.png"), dpi=110)
    print(f"wrote resolution_orientation.png ({len(fig_cids)} markets)")
print("OK D")
