---
title: "IPO Subscriptions — folder guide"
type: index
created: "2026-06-09"
tags: [ipo, subscription, index]
---

# IPO Subscriptions

> Hub: [[POLYMARKET_BRAIN]] · Decision model: [[eu_ipo_broker_subscription_model]]

## Plain-English Summary

Operational tracking for EU retail IPO subscriptions. The **decision model + reasoning** lives in
[[eu_ipo_broker_subscription_model]] (under `notes/overview/market_maps/`, alongside this folder); this
folder holds the **per-deal logs** and the **rolling per-broker fill-rate track record** built from them.

## Files

| file | what it is |
|---|---|
| [[_template]] | Reusable per-deal log template. Copy it per deal; never edit in place. |
| [[SpaceX_SPCX_2026-06]] | The SpaceX (`SPCX`) deal, pre-filled with known prospectus terms. |
| `README.md` | This guide. |

## Add a deal

1. Copy `_template.md` → `<Deal>_<TICKER>_<YYYY-MM>.md` (same folder).
2. Fill the **Prospectus Terms** and your **Per-Broker Plan** *before* the deadline. Get the split from:

   ```bash
   cd polymarket/research
   PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital <C> --tilt 1.0
   ```
3. *After* allocation, fill the **Realised Fill (per broker)** table (`fill_fraction` per broker).
4. Recompute the track record and let it inform the next deal:

   ```bash
   PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --track-record
   PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital <C> --use-track-record
   ```

## How the rolling fill-rate track record works

`--track-record` scans every `*.md` in this folder (skipping `_template.md` and `README.md`), finds the
`Realised Fill (per broker)` table by its header (it keys on the `broker` and `fill_fraction` columns),
and for each broker reports **n deals / mean / sd / min / max** of the realised `fill_fraction`. Rows
still marked `TBD` (or blank with no shares) are **skipped**, not counted as zero — so partially-filled
logs are safe.

Once **every** candidate broker has at least one resolved `fill_fraction`, `--use-track-record`
switches the split from the maturity-prior placeholder to the **measured fill rates**: with flat
pro-rata `shares = fill_rate × subscription`, so the recommender sends more money to whoever fills
you more (the split is proportional to fill rate at `tilt = 1`). It also reports an actual expected
fill `E[F]`. Until then the split rests on the maturity prior as a clearly-labelled placeholder.

## Idempotency

- The tracker only **reads** these notes; running it any number of times changes nothing on disk.
- Logs are **append-only by deal**: one file per deal, copied from the template, never edited in place
  structurally — you only fill in blanks.
- The recommender is a pure function of `(capital, tilt, oversubscribe, broker config, track record)`:
  same inputs → same split, every time.

## Conventions

- `UNKNOWN` = not a vault fact and not provided. Do not guess; confirm before funding.
- Use `TBD` for unresolved post-allocation fields (never `0`, which would read as a real zero fill).
- Keep the `Realised Fill (per broker)` column names exactly as in `_template.md` — the parser depends on them.
