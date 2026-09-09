"""E1 · tokens.parquet — the tree. Reads tokens_raw.parquet + obs_stats.parquet (from E2).
No network. Nothing excluded; verification results are columns. Creates exclusions.csv empty.
Every id stays TEXT. Runs the E1 assertions and prints each with its actual number.
"""
import os
import pandas as pd, numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma")
V1 = os.path.join(HERE, "..", "data", "research_v1"); os.makedirs(V1, exist_ok=True)

tok = pd.read_parquet(os.path.join(DATA, "tokens_raw.parquet"))
obs = pd.read_parquet(os.path.join(V1, "obs_stats.parquet"))
for df in (tok, obs):
    df["asset_id"] = df["asset_id"].astype(str)
tok["condition_id"] = tok["condition_id"].astype(str)
if "event_id" in tok: tok["event_id"] = tok["event_id"].astype("string")

# observation columns come from the built l1/trades (consistent with the tables)
obs = obs.rename(columns={"median_mid": "median_mid_l1", "last_mid": "last_mid_l1", "median_spread": "median_spread_l1"})
t = tok.merge(obs, on="asset_id", how="left")
t["n_l1_events"] = t["n_l1_events"].fillna(0).astype("int64")
t["n_trades"] = t["n_trades"].fillna(0).astype("int64")
t["n_days"] = t["n_days"].fillna(0).astype("int64")
t["median_mid"] = t["median_mid_l1"]; t["last_mid"] = t["last_mid_l1"]
# H0 units fix: obs_stats.median_spread is CENTS ((ask-bid)*100). Store dollars for internal
# consistency with median_mid/last_mid (also dollars), and keep cents alongside for humans.
t["median_spread_cents"] = t["median_spread_l1"]              # cents (0.1 .. 100.0)
t["median_spread"] = t["median_spread_l1"] / 100.0           # dollars, so mid ± spread/2 is valid

# --- resolved_outcome + complement_asset_id, per condition ---
comp = {}; res_out = {}
for cid, g in t.groupby("condition_id"):
    g2 = g.sort_values("outcome_index")
    ids = g2.asset_id.tolist()
    if len(ids) == 2:
        comp[ids[0]] = ids[1]; comp[ids[1]] = ids[0]
    win = g2[g2.is_winner == True]
    res_out[cid] = win.outcome_label.iloc[0] if len(win) == 1 else None
t["complement_asset_id"] = t["asset_id"].map(comp)
t["resolved_outcome"] = t["condition_id"].map(res_out)

# --- check4_status (categorical), computed per condition then applied to both tokens ---
def classify(g):
    if not bool(g.closed.iloc[0]) if pd.notna(g.closed.iloc[0]) else True:
        return "not_resolved"
    if g.is_winner.isna().all():
        return "not_resolved"
    if g.last_mid.isna().any():
        return "no_price"
    w = g[g.is_winner == True]; l = g[g.is_winner == False]
    if len(w) != 1 or len(l) != 1:
        return "not_resolved"
    wl, ll = w.last_mid.iloc[0], l.last_mid.iloc[0]
    if wl >= 0.90 and ll <= 0.10: return "converged_correct"
    if wl <= 0.10 and ll >= 0.90: return "inverted"
    if 0.35 <= wl <= 0.65 and 0.35 <= ll <= 0.65: return "near_half"
    return "no_convergence"
c4 = {cid: classify(g) for cid, g in t.groupby("condition_id")}
t["check4_status"] = t["condition_id"].map(c4)
# unresolved-identity tokens: checks are not applicable
t.loc[t.identity_status == "unresolved", "check4_status"] = None

# --- hours_from_last_seen_to_close ---
ct = pd.to_datetime(t["closed_time"], errors="coerce", utc=True)
close_ms = ct.apply(lambda x: x.value // 10**6 if pd.notna(x) else np.nan)  # pandas 3.0: no Series.view
t["hours_from_last_seen_to_close"] = (close_ms - t["last_seen"]) / 3600000.0

# --- market_slug + path ---
t["market_slug"] = t["slug"]
def mk_path(row):
    if row.identity_status == "unresolved":
        side = "side-a" if row.outcome_index == 0 else "side-b"
        return f"{row.universe}/unknown/{row.condition_id[:10]}/{side}"
    ev = row.event_slug or "unknown-event"
    ms = row.market_slug or row.condition_id[:10]
    lab = row.outcome_label if pd.notna(row.outcome_label) else ("side-a" if row.outcome_index == 0 else "side-b")
    return f"{row.universe}/{ev}/{ms}/{lab}"
t["path"] = t.apply(mk_path, axis=1)
# disambiguate any path collisions with outcome_index (should be rare/none)
dup = t.path.duplicated(keep=False)
if dup.any():
    t.loc[dup, "path"] = t.loc[dup, "path"] + "#" + t.loc[dup, "outcome_index"].astype(str)

cols = ["universe","event_id","event_title","event_slug","event_end_date","neg_risk",
        "condition_id","question","market_slug","outcome_label","outcome_index","outcome",
        "resolved_outcome","closed","closed_time","asset_id","complement_asset_id","identity_status",
        "path","first_seen","last_seen","n_days","n_l1_events","n_trades","median_spread","median_spread_cents","median_mid","last_mid",
        "hours_from_last_seen_to_close",
        "check1_roundtrip","check2_pairing","check3_negrisk","check4_status","check5_indep"]
t = t.reindex(columns=cols).sort_values(["universe","event_slug","market_slug","outcome_index"]).reset_index(drop=True)
t.to_parquet(os.path.join(V1, "tokens.parquet"))

# exclusions.csv — empty, header + comment, stays empty
excl = os.path.join(V1, "exclusions.csv")
if not os.path.exists(excl):
    with open(excl, "w", encoding="utf-8") as f:
        f.write("# exclusions.csv — hand-edited operator instrument, applied at LOAD time only.\n")
        f.write("# Nothing is excluded automatically. Add one asset_id per row with a reason and a date to hide it.\n")
        f.write("asset_id,reason,date\n")

# --- ASSERTIONS ---
print("=== E1 ASSERTIONS ===")
n = len(t); print(f"rows: {n}  (expect 30772)  {'OK' if n==30772 else 'FAIL'}")
u = t.asset_id.is_unique; print(f"asset_id unique: {u}")
vc = t.condition_id.value_counts(); twice = (vc==2).all(); print(f"every condition_id exactly twice: {twice}  (min={vc.min()}, max={vc.max()})")
sym = all(t.set_index('asset_id')['complement_asset_id'].get(b)==a for a,b in zip(t.asset_id, t.complement_asset_id) if pd.notna(b))
print(f"complement symmetric (all pairs): {sym}")
pu = t.path.is_unique; print(f"path unique: {pu}")
nn = t[["universe","condition_id","asset_id","path"]].notna().all().all(); print(f"no nulls in universe/condition_id/asset_id/path: {nn}")
nu = (t.identity_status=='unresolved').sum(); print(f"unresolved count: {nu}  (expect 70)  {'OK' if nu==70 else 'FAIL'}")
pv = t.universe.value_counts().to_dict(); print(f"per-universe rows: {pv}  sum={sum(pv.values())}")
print("check4_status distribution:", t.check4_status.value_counts(dropna=False).to_dict())
print("outcome distribution:", t.outcome.value_counts(dropna=False).to_dict())
fail = not (n==30772 and u and twice and sym and pu and nn and nu==70 and sum(pv.values())==30772)
print("=== RESULT:", "ONE OR MORE ASSERTIONS FAILED — STOP" if fail else "ALL ASSERTIONS PASS", "===")
print("wrote tokens.parquet + exclusions.csv (empty). OK E1")
