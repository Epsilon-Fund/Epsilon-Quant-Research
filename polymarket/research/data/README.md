# polymarket/research/data — artifact tree (gitignored)

> ⚠️ **DO NOT TRUST (2026-07-21, pending regeneration):** `closed_positions.parquet`, `traders.parquet`, `directionality_classification/`, `copyability_candidates/`, `cohorts/`, and `analysis/esports_latency_traders/`. The aggressor of every `_matchOrders` bundle is double-counted (sibling `taker` rows + internal-leg `maker` row) and cross-token bundles fabricate phantom offsetting positions — so PnL/volume **levels** for taker-heavy wallets and all `primary_style`/composition labels are unreliable. Regeneration requires bundle-aware dedup in `scripts/build_closed_positions.py` (a ~multi-hour rebuild over 43 GB of trades — do not run casually; see the spawned task). Evidence: `notes/overview/pm_prealvaro_pipeline_trust_audit_findings.md` Finding 2. Win-rate/bps-style *ratios* are less affected than levels, but verify before use.

Everything under this directory except this README is **gitignored, regenerable or externally-backed data** (Parquet shards, DuckDB files, CSV outputs, cached tapes). Do not look for authoritative documentation here.

**The manifests live in the notes tree** (`polymarket/research/notes/overview/data_quality/`):

- `polymarket_data_manifest.md` — family-level manifest for Parquet/CSV/JSONL/DuckDB/raw artifacts (what exists, where, and how it was produced). Start here.
- `polymarket_csv_output_audit.md` — layout and conventions for generated result/report tables under `data/analysis/csv_outputs/`.
- `polymarket_plot_gallery_index.md` — index of generated figures.
- `polymarket_table_dictionary.md` — definitions for compact column names, bucket labels, and indicators used in output tables.

Invariants (from `brain/CODEX.md`): Parquet shards are **append-only** — never edit in place; DuckDB over Parquet, no DB server; addresses lowercase `0x`-prefixed.
