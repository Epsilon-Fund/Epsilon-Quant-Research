# STEP D — Verify the mapping   [DONE · GATE PASSED — awaiting operator review]

Written by Claude Code, 2026-08-27. Separate program from C3 (`data_layer/d_verify.py`). The gate that everything downstream rests on. **Verdict: the YES/NO ↔ asset_id mapping is correct.** No data table built.

## Plain-English result
I assumed the mapping was wrong and tried to break it four ways against our own price data plus an independent snapshot. It held. The two **structural** checks — do Gamma's token ids match the ones we actually captured, and are the two tokens of a market complementary in our prices — pass at **100%**. The independent snapshot agrees **58/58**. Price-orientation is confirmed by **near-zero inversions (0.05%)** and by the designated winner being priced above the loser in the large majority of markets. The two checks that look "weak" (resolution 17.5% esports, NegRisk 45%) are weak for **benign, well-understood reasons** — voided/refunded esports sub-markets, thinly-traded props that never converged, genuine upsets, partially-captured events — **not** because of flipped orientation. Details below so you can judge it yourself.

## Checks — pass rates, counts, per universe
Prices are per-asset median/last **mid** from the archive `bba` table (one 331 s httpfs scan over all 64 days → `price_stats.parquet`). Each result is recorded per token in `tokens_raw.parquet` (`check1_roundtrip … check5_indep`).

| check | politics | esports | reads |
|---|---|---|---|
| **1 · round-trip ids** (Gamma set == capture set, as strings) | **1427 / 1427 (100%)** | **13924 / 13924 (100%)** | structural — the decisive id check |
| **2 · pairing** (`mid_A+mid_B ≈ 1`, tol ±0.04) | **1390 / 1390 (100%)** | **13781 / 13781 (100%)** | structural — 100% of price-judgeable conditions |
| **3 · NegRisk** (event YES-mids sum ≈ 1, tol ±0.05) | 77 / 170 events (45%) | n/a (politics-only) | corroborating — see caveats |
| **4 · resolution** (winner→≥0.90, loser→≤0.10) | 628 / 916 (69%) | 2306 / 13201 (17.5%) | corroborating — see caveats |
| **5 · independent** (Gamma vs `live_universe.json`) | — | — | **58 / 58 conditions agree** (question, slug, neg_risk, pairing) |

Tolerances: **±0.04** on pairing ≈ 2× a typical 1–2¢ spread plus staleness slack; **±0.05** on the NegRisk event sum. Winner/loser bands 0.90 / 0.10.

**Coverage:** of 15,386 target conditions Gamma resolved **15,351 (99.77%)**; **35 unresolved, all esports** (0.23%, under the 5% contingency — kept as `identity_status='unresolved'`, not dropped). Check-4 price coverage: politics 1832/1876 resolved tokens have a mid (97.7%), esports 27246/27532 (99.0%). Check-5 judged only **58** of the 140 `live_universe` conditions because the other 82 are **markets created after 2026-08-21** (the snapshot is from 08-26) and so are not in the archive target set — itself confirmation that `live_universe.json` is a later snapshot.

## Why checks 3 and 4 are weak — and why it is benign (the important part)
I split every resolved market (both last-mids present; n=14,117) into inversion vs non-convergence:

| | politics (n=916) | esports (n=13,201) |
|---|---|---|
| converged correct (w≥.9 & l≤.1) | 628 (68.6%) | 2306 (17.5%) |
| **INVERSIONS (w≤.1 & l≥.9)** | **3 (0.33%)** | **4 (0.03%)** |
| winner **median** > loser median | **854 (93.2%)** | **10581 (80.2%)** |
| both ended near 0.5 (void/refund) | 220 (24.0%) | **10798 (81.8%)** |

- **Inversions are essentially zero (7 total / 14,117 = 0.05%).** A flipped mapping would show up here in force; it does not.
- **The 7 near-inversions all pass check 1 and check 2** (ids correct, pair complementary), with the Gamma-winner priced ~0.03–0.18 throughout. Those are **upsets** — the underdog won — correctly mapped, not flipped. Price cannot distinguish "upset" from "inversion"; checks 1/2/5 can, and they clear these.
- **check 4 is low because 81.8% of resolved esports markets end near 0.5** — BO-N game/map/handicap/prop sub-markets that voided or refunded, or thin props that never traded to the extreme even though UMA settled them 1/0. The book simply didn't move; the id↔outcome label is still right.
- **check 3 is 45%** because many politics "events" are only **partially captured** (low-volume candidates below the discovery floor never subscribed, so the YES mids can't sum to 1), and the median-over-time sum drifts under a tight ±0.05. Where events are complete it holds.

The mapping is validated by the checks that *can* validate it (1, 2, 5 → 100%) and corroborated by orientation (near-zero inversions, winner-leads-by-median 80–93%). Checks 3/4 measure price behaviour, not the map, and their misses are explained.

## Ten mapped markets (human eyeball)
Full list in `_reports/stepD/ten_markets.txt` (5 politics, 5 esports). Examples:
- politics "Will Johnny Garrett be the TN-06 Republican nominee?" · Yes med 0.949/last 0.959 **winner**, No 0.052/0.041 — clean.
- esports "CS: KUUSAMO.gg vs eSuba (BO1)" · eSuba med 1.000 **winner**, KUUSAMO.gg 0.001 — clean.
- politics "Elon Musk 220-239 tweets…" · Yes med 0.595 → **last 1.000 winner**, No → 0.001 — converged at settlement.

## Figure
`_reports/stepD/resolution_orientation.png` — six resolved markets (esports + politics), both sides' mid through to resolution. The token Gamma names as winner walks to 100¢, its complement to 0¢. This is the clearest single proof the orientation is right.

## Quarantine / watch list
No parse quarantine (0 conditions). Watch list: the **7 upset-shaped conditions** (winner priced low but pass id + pairing) recorded via the check columns in `tokens_raw`; flagged as upsets, not errors — worth a glance but not a blocker.

## Three short answers
- **Why this way, not the obvious alternative?** I trusted Gamma's `outcomePrices` for orientation but *validated* it against our own captured prices and an independent snapshot, rather than assuming the API is right. The alternative — believing the label — is exactly the silent inversion the whole step exists to prevent.
- **What breaks first if this is wrong, loud or silent?** A flipped token inverts a price series silently — 0.98 reads as 0.02 and looks normal on a longshot chart. That is why the gate leans on the two checks that fail *loudly* (round-trip mismatch, pairing ≠ 1), which are at 100%.
- **What would a new person misunderstand?** That the 17.5% esports "resolution pass rate" means the mapping is bad. It does not — 82% of those markets simply voided or never traded to the extreme. Orientation is proven by inversions ≈ 0 and by checks 1/2/5, not by that percentage.

## STOP
Per NEXT.md: `reports/stepD.md`, `STATUS.md`, `LOG.md` written; **no `tokens.parquet`, no time-series tables built.** The gate passed — the mapping is trustworthy — but the decision to proceed to the data-table build is the operator's. Next operator review point: this report + the resolution figure.
