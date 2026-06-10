"""Acceptance tests for the EU retail IPO subscription capital-split recommender.

These pin the load-bearing claims of the decision model:
  (i)    tilt=0 reduces to an EQUAL split (score ignored -> diversify).
  (ii)   with no fill data, the split is weighted by the maturity prior; Revolut (higher
         prior) outweighs the unproven Trade Republic.
  (iii)  THE HEADLINE: once realised fill rates exist, the split steers toward the
         higher-fill broker and is proportional to fill rate at tilt=1.
  (iv)   the safe default (oversubscribe=1.0) makes worst-case funded == capital; an
         oversubscribe>1 raises it and is flagged.
  (v)    broker min/max caps are respected and the residual is reallocated.
  (vi)   the markdown fill-table parser keeps resolved rows, computes blanks, skips TBD.
"""
from __future__ import annotations

from scripts.eu_ipo_capital_split import (
    default_brokers,
    parse_fill_table,
    recommend,
    track_record,
)


def test_tilt_zero_is_equal_weight():
    rec = recommend(default_brokers(), 10000, tilt=0.0)
    ws = [r["weight"] for r in rec.rows]
    assert all(abs(w - ws[0]) < 1e-9 for w in ws)


def test_default_uses_equal_research_prior():
    # both brokers carry the same deal-level research prior (~5%) -> 50/50, E[F] = 0.05*C
    rec = recommend(default_brokers(), 10000, tilt=2.0)
    assert "research-based prior" in rec.weighting_driver
    rev = next(r for r in rec.rows if r["broker"] == "Revolut")
    tr = next(r for r in rec.rows if r["broker"] == "Trade Republic")
    assert abs(rev["weight"] - tr["weight"]) < 1e-9  # equal prior -> equal split
    assert rec.expected_fill is not None and abs(rec.expected_fill - 500.0) < 1e-6


def test_maturity_fallback_when_no_fill_estimate():
    bs = default_brokers()
    for b in bs:
        b.prior_fill_rate = None  # remove the research prior -> fall back to maturity
    rec = recommend(bs, 10000, tilt=2.0)
    assert "maturity prior" in rec.weighting_driver
    rev = next(r for r in rec.rows if r["broker"] == "Revolut")
    tr = next(r for r in rec.rows if r["broker"] == "Trade Republic")
    assert rev["weight"] > tr["weight"]


def test_fill_rate_drives_split_when_known():
    bs = default_brokers()
    by = {b.name: b for b in bs}
    by["Revolut"].realized_fill_rate = 0.30
    by["Trade Republic"].realized_fill_rate = 0.10
    rec = recommend(bs, 10000, tilt=1.0)
    assert "fill-rate" in rec.weighting_driver
    rev = next(r for r in rec.rows if r["broker"] == "Revolut")
    tr = next(r for r in rec.rows if r["broker"] == "Trade Republic")
    # weight proportional to fill rate at tilt=1: 0.30 / (0.30+0.10) = 0.75
    assert abs(rev["weight"] - 0.75) < 1e-9
    assert abs(tr["weight"] - 0.25) < 1e-9
    assert rec.expected_fill is not None and rec.expected_fill > 0


def test_weights_and_budget_close():
    rec = recommend(default_brokers(), 10000, tilt=1.0)
    assert abs(sum(r["weight"] for r in rec.rows) - 1.0) < 1e-9
    assert abs(rec.worst_case_funded - 10000) < 1e-6


def test_oversubscription_raises_worst_case_and_warns():
    rec = recommend(default_brokers(), 10000, tilt=1.0, oversubscribe=1.5)
    assert abs(rec.worst_case_funded - 15000) < 1e-6
    assert any("OVER-SUBSCRIPTION" in w for w in rec.warnings)


def test_max_cap_respected_and_residual_reallocated():
    bs = default_brokers()
    bs[0].max_subscription = 1000.0  # cap Revolut
    rec = recommend(bs, 10000, tilt=2.0)
    rev = next(r for r in rec.rows if r["broker"] == "Revolut")
    assert rev["amount"] <= 1000.0 + 1e-6
    assert abs(sum(r["amount"] for r in rec.rows) - 10000) < 1e-6


def test_revolut_min_binds_on_small_budget():
    rec = recommend(default_brokers(), 600, tilt=2.0)
    rev = next(r for r in rec.rows if r["broker"] == "Revolut")
    assert rev["amount"] >= 500.0 - 1e-6  # $500 Revolut minimum


def test_fill_table_parser():
    sample = (
        "| broker | requested_shares | filled_shares | fill_fraction | effective_price | notes |\n"
        "|---|---|---|---|---|---|\n"
        "| Revolut | 100 | 25 | 0.25 | 135 | ok |\n"
        "| Trade Republic | 100 | 10 |  | 135 | computed |\n"
        "| DEGIRO | 100 |  | TBD | 135 | unresolved |\n"
    )
    got = {r["broker"]: round(r["fill_fraction"], 3) for r in parse_fill_table(sample)}
    assert got == {"Revolut": 0.25, "Trade Republic": 0.1}


def test_track_record_empty_when_no_logs(tmp_path):
    assert track_record(tmp_path) == {}
