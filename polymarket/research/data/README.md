# polymarket/research/data — artifact tree (gitignored)

Everything under this directory except this README is **gitignored, regenerable or externally-backed data** (Parquet shards, DuckDB files, CSV outputs, cached tapes). Do not look for authoritative documentation here.

**The manifests live in the notes tree** (`polymarket/research/notes/overview/data_quality/`):

- `polymarket_data_manifest.md` — family-level manifest for Parquet/CSV/JSONL/DuckDB/raw artifacts (what exists, where, and how it was produced). Start here.
- `polymarket_csv_output_audit.md` — layout and conventions for generated result/report tables under `data/analysis/csv_outputs/`.
- `polymarket_plot_gallery_index.md` — index of generated figures.
- `polymarket_table_dictionary.md` — definitions for compact column names, bucket labels, and indicators used in output tables.

Invariants (from `brain/CODEX.md`): Parquet shards are **append-only** — never edit in place; DuckDB over Parquet, no DB server; addresses lowercase `0x`-prefixed.
