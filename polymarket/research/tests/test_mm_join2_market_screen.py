"""Tests for the Join-2c 5-screen politics-NegRisk market selection (offline, no network).

Gamma rows are fixtures; the historical caches are synthetic parquet files built in-test,
so the duckdb loaders are exercised for real. No test touches the network.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from scripts.mm_join2_market_screen import (
    HEADROOM_MIN,
    Candidate,
    classify_bucket,
    fetch_active_negrisk_markets,
    load_directional_scores,
    load_headroom_and_top3,
    render_table,
    screen_markets,
)

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------
# screen 2 — bucket classifier
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_bucket,expected_scope", [
    ("Will Zohran win the 2028 presidential election?", "us_2028_outrights", False),
    ("2028 Democratic presidential nominee?", "us_2028_outrights", False),
    ("Will the GOP hold the Senate in the 2026 midterm elections us?", "us_2026_races", True),
    ("Who wins the 2026 Irish presidential election? irish-presidential-election",
     "non_us_elections", True),
    ("Will Trump fire the Fed chair this year?", "trump_personnel_policy", True),
    ("Will the Senate confirm the nominee for Secretary of State?",
     "trump_personnel_policy", True),
    ("Will the government shutdown end before August?", "other_politics", True),
    ("Will Bitcoin close above 100k?", "not_politics", False),
])
def test_classify_bucket(text: str, expected_bucket: str, expected_scope: bool) -> None:
    bucket, in_scope = classify_bucket(text.lower())
    assert bucket == expected_bucket
    assert in_scope is expected_scope


# --------------------------------------------------------------------------------------
# screens 3+4 — duckdb loaders on synthetic parquet
# --------------------------------------------------------------------------------------

@pytest.fixture()
def caches(tmp_path: Path) -> tuple[Path, Path]:
    carry = tmp_path / "carry.parquet"
    con = duckdb.connect()
    # market 111: top3 = a1,a2,a3 (90 USD), non-top3 = a4 (10 USD) → headroom 10%
    # market 222: top3 = b1,b2,b3 (99 USD), non-top3 = b4 (1 USD)  → headroom 1% (FAIL)
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES
                (111, '0xa1', 50.0), (111, '0xa2', 25.0), (111, '0xa3', 15.0), (111, '0xa4', 10.0),
                (222, '0xb1', 60.0), (222, '0xb2', 30.0), (222, '0xb3', 9.0),  (222, '0xb4', 1.0)
            ) AS t(market_id, address, maker_usd)
        ) TO '{carry.as_posix()}' (FORMAT PARQUET)
    """)
    direction = tmp_path / "direction.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES
                ('0xa1', 0.10), ('0xa2', 0.20), ('0xa3', 0.30),
                ('0xb1', 0.90), ('0xb2', 0.80), ('0xb3', 0.70)
            ) AS t(address, pct_markets_two_sided_directional_vw)
        ) TO '{direction.as_posix()}' (FORMAT PARQUET)
    """)
    return carry, direction


def test_headroom_and_top3_from_parquet(caches: tuple[Path, Path]) -> None:
    carry, _ = caches
    headroom, top3 = load_headroom_and_top3(carry)
    assert headroom["111"] == pytest.approx(0.10)
    assert headroom["222"] == pytest.approx(0.01)
    assert set(top3["111"]) == {"0xa1", "0xa2", "0xa3"}


def test_directional_scores_from_parquet(caches: tuple[Path, Path]) -> None:
    _, direction = caches
    scores = load_directional_scores({"0xa1", "0xb1", "0xmissing"}, direction)
    assert scores["0xa1"] == pytest.approx(0.10)
    assert scores["0xb1"] == pytest.approx(0.90)
    assert "0xmissing" not in scores


# --------------------------------------------------------------------------------------
# the combined screen
# --------------------------------------------------------------------------------------

def _gamma_row(gid: str, cond: str, question: str, volume: float = 50_000.0,
               end: str = "2026-10-01T00:00:00Z") -> dict:
    return {
        "id": gid, "conditionId": cond, "question": question,
        "slug": question.lower().replace(" ", "-")[:50],
        "negRisk": True, "acceptingOrders": True,
        "volumeNum": volume, "endDate": end,
        "events": [{"title": question}],
    }


def test_screen_markets_applies_all_five_screens(caches: tuple[Path, Path]) -> None:
    carry, direction = caches
    headroom, top3 = load_headroom_and_top3(carry)
    all_top3 = {a for addrs in top3.values() for a in addrs}
    scores = load_directional_scores(all_top3, direction)

    rows = [
        # in-scope, history says 10% headroom (pass), uninformed top-3 (a* scores low)
        _gamma_row("111", "0xc1", "Who wins the 2026 Irish presidential election?"),
        # in-scope but history says 1% headroom → hard FAIL, dropped
        _gamma_row("222", "0xc2", "Will Trump fire the Fed chair?"),
        # in-scope, NO history → screen 3 UNKNOWN, ranks below the known pass
        _gamma_row("333", "0xc3", "Will the government shutdown end before August?"),
        # 2028 outright → excluded by screen 2
        _gamma_row("444", "0xc4", "2028 presidential election winner?"),
        # volume floor
        _gamma_row("555", "0xc5", "Will parliament pass the French budget bill?", volume=100.0),
        # end date beyond the horizon → clarity flag off but not dropped
        _gamma_row("666", "0xc6", "Who wins the 2027 French presidential election?",
                   end="2027-12-01T00:00:00Z"),
    ]
    cands = screen_markets(rows, headroom, top3, scores, now_utc=NOW)

    ids = [c.gamma_id for c in cands]
    assert "222" not in ids                      # screen-3 hard fail
    assert "444" not in ids                      # 2028 excluded
    assert "555" not in ids                      # volume floor
    assert ids[0] == "111"                       # known-pass + uninformed-pref ranks first
    first = cands[0]
    assert first.headroom_pass is True and first.uninformed_flow_pref is True
    assert first.resolution_clarity is True
    unknown = next(c for c in cands if c.gamma_id == "333")
    assert unknown.headroom_pass is None         # UNKNOWN, not a fabricated pass
    horizon = next(c for c in cands if c.gamma_id == "666")
    assert horizon.resolution_clarity is False

    table = render_table(cands)
    assert "irish" in table.lower() or "Irish" in table


def test_fetch_filters_negrisk_and_paginates() -> None:
    pages: list[str] = []

    def fake_get(url: str):
        pages.append(url)
        if "offset=0" in url:
            return [                                       # a FULL page → fetch continues
                {"id": "1", "negRisk": True, "acceptingOrders": True},
                {"id": "2", "negRisk": False},
                {"id": "3"},                              # no flag → not negRisk
            ]
        return []                                          # second page empty → stop

    rows = fetch_active_negrisk_markets(pages=5, page_size=3, get_json=fake_get)
    assert [r["id"] for r in rows] == ["1"]
    assert len(pages) == 2                                 # stopped after the empty page

    # a short page also stops the walk (no pointless extra request)
    pages.clear()
    rows = fetch_active_negrisk_markets(pages=5, page_size=100, get_json=fake_get)
    assert [r["id"] for r in rows] == ["1"]
    assert len(pages) == 1


def test_candidate_row_serialization_handles_unknowns() -> None:
    c = Candidate(condition_id="0xc", gamma_id="9", question="q", bucket="other_politics",
                  volume_usd=1.0, end_date="")
    row = c.as_row()
    assert row["headroom_pass"] == "UNKNOWN"
    assert row["uninformed_flow_pref"] == "UNKNOWN"
    assert HEADROOM_MIN == 0.05                            # the pre-registered 5% bar
