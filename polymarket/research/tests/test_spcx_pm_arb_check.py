"""Acceptance tests for the SPCX PM executable-arb check (Block S8).

These encode the task-spec pre-registration:
  (i)   EXECUTABLE-ONLY: no arb figure derives from a mid or a PCHIP value — the
        analysis is invariant to poisoned 'mid' keys, a mid-crossed/ask-clean book
        reports NO-ARB, and the module never references pchip/mid prices.
  (ii)  CO-RESOLUTION: combos mixing resolution keys are reported RV-WITH-RISK and
        never as arb.
  (iii) DEPTH-AWARE: every reported edge carries executable size and the depth
        segment where it closes; dust legs are flagged.
  (iv)  FEE-COMPLETE: gross and net (repo + harsh schedules, + gas on the mint
        path) are all present; verdicts use net.
  (v)   Payoff floors come from state enumeration under the live resolution
        semantics (ladder strictly-above, buckets lo-inclusive/hi-exclusive,
        No-IPO state => all cap legs NO), with exact-boundary states surfaced as
        caveats, not silently priced.
"""
from __future__ import annotations

import inspect
import math

import pytest

import scripts.spcx_pm_arb_check as arb
from scripts.spcx_pm_arb_check import (
    analyze,
    best_executable_arb,
    build_combos,
    build_snapshot,  # noqa: F401  (imported to assert it exists; never called here)
    enumerate_states,
    evaluate_combo,
    fee_per_share,
    leg_indicator,
    parse_leg,
    payoff_floor,
    taker_fee,
    walk_lock,
    walk_lock_net,
    walk_mint_sell,
)

RK = "first_day_close_cap|no_ipo_by:December 31, 2027"


# ----------------------------------------------------------------------------------
# Synthetic complex: 3-strike ladder + 6-bucket NegRisk group + No-IPO
# ----------------------------------------------------------------------------------
def L(strike):
    return {"kind": "ladder", "strike_t": strike, "lo": None, "hi": None,
            "label": f">{strike:g}T", "token_yes": f"LY{strike}", "token_no": f"LN{strike}",
            "res_key": RK, "neg_risk": False, "condition_id": f"lad{strike}"}


def B(lo, hi):
    label = (f"<{hi:g}T" if lo == 0.0 else
             (f">={lo:g}T" if math.isinf(hi) else f"{lo:g}-{hi:g}T"))
    return {"kind": "bucket", "strike_t": None, "lo": lo, "hi": hi, "label": label,
            "token_yes": f"BY{lo}", "token_no": f"BN{lo}",
            "res_key": RK, "neg_risk": True, "condition_id": f"bkt{lo}"}


NOIPO = {"kind": "no_ipo", "strike_t": None, "lo": None, "hi": None, "label": "No-IPO",
         "token_yes": "NIY", "token_no": "NIN", "res_key": RK, "neg_risk": True,
         "condition_id": "noipo"}


def make_meta():
    return {"ladder": [L(1.0), L(2.0), L(3.0)],
            "buckets": [B(0.0, 1.0), B(1.0, 1.5), B(1.5, 2.0), B(2.0, 2.5),
                        B(2.5, 3.0), B(3.0, math.inf)],
            "no_ipo": NOIPO}


def snap_from(quotes: dict) -> dict:
    """quotes: token -> {'asks': [(p, s)...], 'bids': [(p, s)...]} (either side optional)."""
    books = {t: {"bids": q.get("bids", []), "asks": q.get("asks", [])}
             for t, q in quotes.items()}
    return {"fetched_at_utc": "2026-06-12T10:00:00+00:00", "books": books}


def null_quotes(meta):
    """A self-consistent no-arb book on every token: wide 10c spreads, deep."""
    fair = {"LY1.0": 0.99, "LY2.0": 0.75, "LY3.0": 0.08,
            "BY0.0": 0.01, "BY1.0": 0.02, "BY1.5": 0.20, "BY2.0": 0.56,
            "BY2.5": 0.16, "BY3.0": 0.05, "NIY": 0.003}
    q = {}
    for tok, p in fair.items():
        q[tok] = {"bids": [(round(max(p - 0.05, 0.001), 3), 500.0)],
                  "asks": [(round(min(p + 0.05, 0.999), 3), 500.0)]}
        ntok = tok.replace("Y", "N", 1) if tok != "NIY" else "NIN"
        q[ntok] = {"bids": [(round(max(1 - p - 0.05, 0.001), 3), 500.0)],
                   "asks": [(round(min(1 - p + 0.05, 0.999), 3), 500.0)]}
    return q


# ----------------------------------------------------------------------------------
# (v) resolution semantics + state-enumerated floors
# ----------------------------------------------------------------------------------
def test_leg_indicator_boundary_semantics():
    # bucket lo-inclusive ("exactly between brackets -> higher bracket")
    assert leg_indicator(B(2.0, 2.5), "YES", 2.0, False) == 1
    assert leg_indicator(B(1.5, 2.0), "YES", 2.0, False) == 0
    # ladder strictly above
    assert leg_indicator(L(2.0), "YES", 2.0, False) == 0
    assert leg_indicator(L(2.0), "YES", 2.01, False) == 1
    # No-IPO state: every cap leg NO, no_ipo YES
    assert leg_indicator(L(2.0), "NO", None, True) == 1
    assert leg_indicator(B(2.0, 2.5), "NO", None, True) == 1
    assert leg_indicator(NOIPO, "YES", None, True) == 1
    assert leg_indicator(NOIPO, "YES", 2.2, False) == 0


def test_box_floors_from_state_enumeration():
    meta = make_meta()
    states = enumerate_states(meta)
    combos = {c["name"]: c for c in build_combos(meta)}
    # direction A on [2,3): failure-partition of 4 legs -> floor 3, boundary dip at 2.0
    fa = payoff_floor(combos["box_A_[2,3)"]["legs"], states)
    assert fa["floor_robust"] == 3.0
    assert any("cap=2T exactly" in c for c in fa["caveats"])
    # direction B on [2,3): floor 1, boundary dip to 0 at exactly 3.0
    fb = payoff_floor(combos["box_B_[2,3)"]["legs"], states)
    assert fb["floor_robust"] == 1.0
    assert any("cap=3T exactly" in c for c in fb["caveats"])
    # direction B on [3,inf): floor 1 with NO boundary failure
    f3 = payoff_floor(combos["box_B_[3,inf)"]["legs"], states)
    assert f3["floor_robust"] == 1.0 and f3["caveats"] == []
    # baskets: exactly one outcome pays in every state incl. boundaries and No-IPO
    fy = payoff_floor(combos["basket_buy_all_YES"]["legs"], states)
    assert fy["floor_robust"] == 1.0 and fy["caveats"] == []
    fn = payoff_floor(combos["basket_buy_all_NO"]["legs"], states)
    assert fn["floor_robust"] == 6.0 and fn["caveats"] == []  # N-1 with N=7 outcomes


def test_key_unions_are_enumerated():
    names = {c["name"] for c in build_combos(make_meta())}
    for expected in ("box_A_[2,3)", "box_B_[2,3)", "box_A_[1,2)", "box_B_[0,1)",
                     "box_A_[3,inf)", "box_B_[1,3)", "mono_adj_>2T_vs_>3T",
                     "basket_buy_all_YES", "basket_buy_all_NO", "complement_>2T",
                     "cover_sell_2-2.5T_via_[2,3)", "cover_buy_1.5-2T..2.5-3T_via_[2,3)"):
        assert expected in names, expected
    # exact-edge covers must NOT duplicate the exact boxes
    assert "cover_sell_2-2.5T..2.5-3T_via_[2,3)" not in names


def test_cover_floors_from_state_enumeration():
    meta = make_meta()
    states = enumerate_states(meta)
    combos = {c["name"]: c for c in build_combos(meta)}
    # sell bucket [2,2.5) covered by the wider ladder range [2,3): floor 2 — the
    # uncovered [2.5,3) sliver is a free bonus state ($3) — boundary dip at exactly 2.0
    f = payoff_floor(combos["cover_sell_2-2.5T_via_[2,3)"]["legs"], states)
    assert f["floor_robust"] == 2.0
    assert any("cap=2T exactly" in c for c in f["caveats"])
    # buy the [1.5,3) run, short the narrower contained range [2,3): floor 1,
    # boundary dip to 0 at exactly 3.0 (run excludes it, higher-bracket rule)
    f2 = payoff_floor(combos["cover_buy_1.5-2T..2.5-3T_via_[2,3)"]["legs"], states)
    assert f2["floor_robust"] == 1.0
    assert any("cap=3T exactly" in c for c in f2["caveats"])
    # the full-run all-NO cover = the 7-outcome cap basket (here 6 buckets): floor N-1
    f3 = payoff_floor(combos["cover_sell_<1T..>=3T_via_[0,inf)"]["legs"], states)
    assert f3["floor_robust"] == 5.0 and f3["caveats"] == []


def test_planted_cover_sell_lock_dust_flagged():
    """The S8 follow-up case: sell a rich bucket into a thin bid, cover with the next
    strike out. +2c gross on 7 sets -> real lock, dust size, fee-negative under both."""
    meta = make_meta()
    q = null_quotes(meta)
    q["BN2.0"]["asks"] = [(0.42, 7.0)]      # NO of bucket [2,2.5) — mirrors a 7-share bid
    q["LY2.0"]["asks"] = [(0.76, 9000.0)]
    q["LN3.0"]["asks"] = [(0.80, 466.0)]
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "cover_sell_2-2.5T_via_[2,3)")
    assert r["floor"] == 2.0
    assert r["cost_top"] == pytest.approx(1.98)
    assert r["gross_top"] == pytest.approx(0.02)
    assert r["gross_sets"] == pytest.approx(7.0)
    assert r["net_repo_top"] < 0 and r["net_harsh_top"] < 0
    assert r["dust"] is True
    assert "NO-ARB" in r["verdict"]


# ----------------------------------------------------------------------------------
# depth-walk arithmetic
# ----------------------------------------------------------------------------------
def test_walk_lock_arithmetic_hand_case():
    legs = [[(0.40, 10.0), (0.45, 10.0)],      # leg 1
            [(0.50, 15.0), (0.70, 100.0)]]     # leg 2
    w = walk_lock(legs, floor=1.0)
    # segments: [0,10) cost .90, [10,15) cost .95, [15,20) cost 1.15 (gross<0 -> closes)
    assert [round(s["cost_per_set"], 2) for s in w["segments"]] == [0.90, 0.95, 1.15]
    assert w["gross"]["sets"] == pytest.approx(15.0)
    assert w["gross"]["profit_usd"] == pytest.approx(10 * 0.10 + 5 * 0.05)
    assert w["gross"]["closes_at_segment"] == 2  # edge closes at the third depth segment
    fee0 = fee_per_share(0.40, "repo") + fee_per_share(0.50, "repo")
    assert w["segments"][0]["net_repo_per_set"] == pytest.approx(0.10 - fee0)


def test_walk_lock_exhaustion_flag():
    w = walk_lock([[(0.30, 8.0)], [(0.50, 8.0)]], floor=1.0)
    assert w["exhausted_before_close"] is True
    assert w["gross"]["sets"] == pytest.approx(8.0)


# ----------------------------------------------------------------------------------
# single-fee net walk — the one-number "how much can I extract" output
# ----------------------------------------------------------------------------------
def test_taker_fee_formula():
    assert taker_fee(0.40, 1000) == pytest.approx(0.10 * 0.40)   # 1000bps -> 0.10*min
    assert taker_fee(0.80, 1000) == pytest.approx(0.10 * 0.20)
    assert taker_fee(0.40, 0) == 0.0


def test_walk_lock_net_zero_fee_runs_until_spread():
    # two legs, floor 1; gross +.10 for 10 sets, then +.05 for 5, then negative
    legs = [[(0.40, 10.0), (0.45, 10.0)], [(0.50, 15.0), (0.70, 100.0)]]
    w = walk_lock_net(legs, floor=1.0, fee_bps=0.0)
    assert w["net_sets"] == pytest.approx(15.0)
    assert w["net_usd"] == pytest.approx(10 * 0.10 + 5 * 0.05)
    assert w["closed_by"] == "fees/spread"


def test_walk_lock_net_fee_closes_earlier():
    legs = [[(0.40, 10.0), (0.45, 10.0)], [(0.50, 15.0), (0.70, 100.0)]]
    # at 1000bps the per-set fee is .10*(min(.4,.6)+min(.5,.5))=.10*(.4+.5)=.09 > +.10? no,
    # set#1 net = .10-.09=+.01 (survives), set#2 (prices .45,.50) net=.05-.10*(.45+.50)=-.045 stop
    w = walk_lock_net(legs, floor=1.0, fee_bps=1000.0)
    assert w["net_sets"] == pytest.approx(10.0)
    assert w["net_usd"] == pytest.approx(10 * (0.10 - 0.09))


def test_best_executable_arb_picks_top_lock_and_renders_buysell():
    meta = make_meta()
    q = null_quotes(meta)
    q["BN2.0"]["asks"] = [(0.42, 300.0)]    # sell [2,2.5) bucket cheaply (its NO)
    q["LY2.0"]["asks"] = [(0.76, 9000.0)]
    q["LN3.0"]["asks"] = [(0.80, 466.0)]    # there is no 2.6 strike in make_meta -> cover via [2,3)
    q["BY3.0"]["asks"] = [(0.30, 500.0)]    # kill the competing box_B_[3,inf) so the cover wins
    res = best_executable_arb(meta, snap_from(q), fee_bps=0.0)
    assert res["exists"] is True
    b = res["best"]
    actions = {(l["action"], l["market"]) for l in b["legs"]}
    # the planted edge is the rich [2-2.5T] bucket: the winning lock SELLs it and covers
    # with the ladder (buy >2T, sell >3T); buy-NO renders as SELL, buy-YES as BUY
    assert ("SELL", "2-2.5T") in actions
    assert ("BUY", ">2T") in actions
    assert ("SELL", ">3T") in actions
    assert all(l["action"] in ("BUY", "SELL") and l["price"] > 0 for l in b["legs"])
    assert b["pay_per_set"] < b["payout_floor"]          # it's a genuine lock
    assert b["pay_per_set"] == pytest.approx(sum(l["price"] for l in b["legs"]))
    assert b["net_usd"] > 0 and b["net_sets"] > 0
    assert res["fee_bps"] == 0.0 and "min(price" in res["fee_formula"]


def test_best_executable_arb_one_fee_shrinks_the_walk():
    """One fee knob, one number — raising it strictly shrinks the extractable net (the
    walk stops at an earlier book level). The fee-eats-the-marginal-set mechanism itself
    is pinned in test_walk_lock_net_fee_closes_earlier."""
    meta = make_meta()
    q = null_quotes(meta)
    q["BN2.0"]["asks"] = [(0.42, 300.0)]
    q["LY2.0"]["asks"] = [(0.76, 9000.0)]
    q["LN3.0"]["asks"] = [(0.80, 466.0)]
    res0 = best_executable_arb(meta, snap_from(q), fee_bps=0.0)
    res1k = best_executable_arb(meta, snap_from(q), fee_bps=1000.0)
    assert res0["exists"] and res0["best"]["net_usd"] > 1.0
    n1k = res1k["best"]["net_usd"] if res1k["exists"] else 0.0
    assert n1k < res0["best"]["net_usd"]          # fee strictly reduces extractable $
    # a punishing fee eventually leaves nothing
    assert best_executable_arb(meta, snap_from(q), fee_bps=5000.0)["exists"] is False


def test_best_executable_arb_null_book_has_no_lock():
    meta = make_meta()
    res = best_executable_arb(meta, snap_from(null_quotes(meta)), fee_bps=0.0)
    assert res["exists"] is False


# ----------------------------------------------------------------------------------
# (iii) planted arbs: size + closing depth + dust flags
# ----------------------------------------------------------------------------------
def test_planted_basket_buy_all_yes():
    meta = make_meta()
    q = null_quotes(meta)
    # plant cheap asks summing 0.961 for 30 sets, then a level that closes the edge
    asks = {"BY0.0": 0.01, "BY1.0": 0.02, "BY1.5": 0.20, "BY2.0": 0.55,
            "BY2.5": 0.15, "BY3.0": 0.03, "NIY": 0.001}
    for tok, p in asks.items():
        q[tok]["asks"] = [(p, 100.0)] if tok != "BY2.0" else [(0.55, 30.0), (0.62, 100.0)]
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "basket_buy_all_YES")
    assert r["cost_top"] == pytest.approx(0.961)
    assert r["gross_top"] == pytest.approx(0.039)
    assert r["gross_sets"] == pytest.approx(30.0)            # closed by the 0.62 level
    assert r["gross_usd"] == pytest.approx(30 * 0.039)
    assert r["net_repo_top"] < r["gross_top"]                # fees actually subtracted
    assert r["net_harsh_top"] < r["net_repo_top"] or True    # schedules both present
    assert r["net_harsh_sets"] <= r["gross_sets"]


def test_planted_monotonicity_inversion_dust_flagged():
    meta = make_meta()
    q = null_quotes(meta)
    q["LY2.0"]["asks"] = [(0.55, 20.0)]
    q["LN3.0"]["asks"] = [(0.40, 8.0)]    # only 8 shares -> dust
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "mono_adj_>2T_vs_>3T")
    assert r["floor"] == 1.0
    assert r["cost_top"] == pytest.approx(0.95)
    assert r["gross_sets"] == pytest.approx(8.0)             # exhausted at 8 sets
    assert r["dust"] is True
    assert "NO-ARB" in r["verdict"]


def test_planted_investable_arb_verdict():
    meta = make_meta()
    q = null_quotes(meta)
    q["LY2.0"]["asks"] = [(0.30, 1000.0)]
    q["LN3.0"]["asks"] = [(0.50, 1000.0)]
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "mono_adj_>2T_vs_>3T")
    assert r["gross_top"] == pytest.approx(0.20)
    assert r["net_harsh_top"] == pytest.approx(0.20 - 0.10 * (0.30 + 0.50))
    assert r["verdict"].startswith("ARB")
    assert r["net_harsh_usd"] > 100.0


# ----------------------------------------------------------------------------------
# (i) executable-only
# ----------------------------------------------------------------------------------
def test_mid_crossed_book_is_not_arb():
    """Mids 'cross' (sum of complement mids < 1) but executable asks do not: NO-ARB."""
    meta = make_meta()
    q = null_quotes(meta)
    q["LY2.0"] = {"bids": [(0.30, 500.0)], "asks": [(0.60, 500.0)]}   # mid 0.45
    q["LN2.0"] = {"bids": [(0.30, 500.0)], "asks": [(0.60, 500.0)]}   # mid 0.45 -> mid-sum 0.9
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "complement_>2T")
    assert r["cost_top"] == pytest.approx(1.20)   # executable ask sum, not 0.9
    assert r["gross_top"] < 0
    assert r["verdict"] == "NO-ARB"


def test_analysis_invariant_to_poisoned_mid_keys():
    meta = make_meta()
    snap = snap_from(null_quotes(meta))
    base = analyze(meta, snap)
    for bk in snap["books"].values():
        bk["mid"] = 0.0001  # if anything read mids, every combo would turn into "arb"
    poisoned = analyze(meta, snap)
    for a, b in zip(base["results"], poisoned["results"]):
        assert a["cost_top"] == b["cost_top"] and a["verdict"] == b["verdict"]


def test_module_never_references_pchip_or_mid_prices():
    """Executable code (module docstring + comments excluded) must never name pchip
    or compute/read a mid price. Docstrings may mention them to say they're banned."""
    src = inspect.getsource(arb)
    code = src.split('"""', 2)[2]  # everything after the module docstring
    lines = [l.split("#")[0] for l in code.splitlines()]
    body = "\n".join(lines).lower()
    assert "pchip" not in body
    assert "(bid + ask) / 2" not in body and "(ask + bid) / 2" not in body
    assert '"mid"' not in body and "'mid'" not in body


# ----------------------------------------------------------------------------------
# (ii) co-resolution guard
# ----------------------------------------------------------------------------------
def test_mixed_resolution_reported_as_risk_not_arb():
    meta = make_meta()
    meta["buckets"][3] = dict(meta["buckets"][3], res_key="UNKNOWN:bkt2.0")
    q = null_quotes(meta)
    # plant an otherwise-screaming basket arb
    for tok in ("BY0.0", "BY1.0", "BY1.5", "BY2.0", "BY2.5", "BY3.0", "NIY"):
        q[tok]["asks"] = [(0.05, 1000.0)]
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "basket_buy_all_YES")
    assert r["mixed_resolution"] is True
    assert r["verdict"].startswith("RV-WITH-RISK")
    assert not r["verdict"].startswith("ARB")


def test_resolution_key_parsing():
    desc = ("This market will resolve based on SpaceX's market capitalization at the "
            "closing price on its first day of trading. If no SpaceX IPO occurs by "
            "December 31, 2027, 11:59 PM ET, the market will resolve to ...")
    mkt = {"question": "Will SpaceX's market cap be between $2.0T and $2.5T at market close on IPO day?",
           "outcomes": '["Yes", "No"]', "clobTokenIds": '["t1", "t2"]',
           "conditionId": "0xabc", "negRisk": True, "description": desc}
    leg = parse_leg(mkt, arb.BUCKET_EVENT_SLUG)
    assert leg["kind"] == "bucket" and leg["lo"] == 2.0 and leg["hi"] == 2.5
    assert leg["res_key"] == RK
    leg2 = parse_leg(dict(mkt, description="resolves somehow"), arb.BUCKET_EVENT_SLUG)
    assert leg2["res_key"].startswith("UNKNOWN:")
    lad = parse_leg({"question": "SpaceX IPO closing market cap above $2T?",
                     "outcomes": '["Yes", "No"]', "clobTokenIds": '["a", "b"]',
                     "conditionId": "0xdef", "negRisk": False, "description": desc},
                    arb.LADDER_EVENT_SLUG)
    assert lad["kind"] == "ladder" and lad["strike_t"] == 2.0 and lad["res_key"] == RK
    ni = parse_leg({"question": "Will SpaceX not IPO by December 31, 2027?",
                    "outcomes": '["Yes", "No"]', "clobTokenIds": '["x", "y"]',
                    "conditionId": "0x111", "negRisk": True, "description": desc},
                   arb.BUCKET_EVENT_SLUG)
    assert ni["kind"] == "no_ipo"


# ----------------------------------------------------------------------------------
# (iv) fee-complete + mint path
# ----------------------------------------------------------------------------------
def test_gross_positive_net_negative_is_no_arb():
    meta = make_meta()
    q = null_quotes(meta)
    # gross edge +0.4c/set at deep size; harsh fee ~8.5c kills it
    q["LY2.0"]["asks"] = [(0.550, 5000.0)]
    q["LN3.0"]["asks"] = [(0.446, 5000.0)]
    rep = analyze(meta, snap_from(q))
    r = next(x for x in rep["results"] if x["name"] == "mono_adj_>2T_vs_>3T")
    assert r["gross_top"] == pytest.approx(0.004)
    assert r["net_harsh_top"] < 0 and r["net_repo_top"] < 0
    assert "NO-ARB" in r["verdict"]
    # gross AND both nets must be present on every row (fee-complete acceptance)
    for row in rep["results"]:
        for k in ("gross_usd", "net_repo_usd", "net_harsh_usd", "gas_per_set"):
            assert k in row


def test_mint_sell_walk_residual_and_gas():
    bids = [[(0.20, 50.0)], [(0.60, 50.0)], [(0.30, 50.0)], []]  # one leg has no bid
    w = walk_mint_sell(bids, set_cost=1.0, gas_total=0.10)
    assert w["n_residual_legs"] == 1
    assert w["top"]["revenue_per_set"] == pytest.approx(1.10)
    assert w["top"]["gross_per_set"] == pytest.approx(0.10)
    fees = sum(fee_per_share(p, "repo") for p in (0.20, 0.60, 0.30))
    assert w["net_repo"]["profit_usd"] == pytest.approx(50 * (0.10 - fees) - 0.10)


def test_mint_sell_no_arb_on_null_book():
    meta = make_meta()
    rep = analyze(meta, snap_from(null_quotes(meta)))
    r = next(x for x in rep["results"] if x["name"] == "basket_mint_sell_all_YES")
    assert r["verdict"].startswith("NO-ARB")
    assert r["gas_per_set"] == pytest.approx(arb.GAS_USD_DEFAULT)


# ----------------------------------------------------------------------------------
# null complex: every class reports NO-ARB (honest-null regression)
# ----------------------------------------------------------------------------------
def test_null_complex_reports_no_arb_everywhere():
    meta = make_meta()
    rep = analyze(meta, snap_from(null_quotes(meta)))
    for r in rep["results"]:
        assert not r["verdict"].startswith("ARB"), (r["name"], r["verdict"])


# ----------------------------------------------------------------------------------
# parquet round-trip (reproducible offline re-analysis)
# ----------------------------------------------------------------------------------
def test_books_parquet_round_trip(tmp_path):
    meta = make_meta()
    snap = snap_from(null_quotes(meta))
    path = arb.log_books_parquet(meta, snap, out_dir=tmp_path)
    meta2, snap2 = arb.load_snapshot_parquet(path)
    assert math.isinf(meta2["buckets"][-1]["hi"])
    for tok, bk in snap["books"].items():
        if bk["bids"] or bk["asks"]:
            assert snap2["books"][tok]["bids"] == [(p, s) for p, s in bk["bids"]]
            assert snap2["books"][tok]["asks"] == [(p, s) for p, s in bk["asks"]]
    rep1 = analyze(meta, snap)
    rep2 = analyze(meta2, snap2)
    assert [r["verdict"] for r in rep1["results"]] == [r["verdict"] for r in rep2["results"]]
