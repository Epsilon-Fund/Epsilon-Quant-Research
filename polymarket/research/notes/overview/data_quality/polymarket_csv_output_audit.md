# Polymarket CSV Output Audit

> Hub: [[POLYMARKET_BRAIN]] / [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Headline

Polymarket CSV outputs are now organized by research cluster instead of being spread across the root of `polymarket/research/data/analysis/`. The cleanup makes generated result tables easier to trace back to their notes and keeps future Codex/Cowork sessions from treating one flat folder as a mystery pile.

For the current attachment-level CSV wikilinks and the broader Parquet/JSONL/DuckDB family map, see [[polymarket_data_manifest]].

## What Was Audited

This pass checked project CSV files under `polymarket/`, excluding dependency sample/test CSVs inside `.venv`. The focus was report/result CSVs created by markdown-driven research notes and scripts.

The unit of observation in the table below is a filesystem bucket. `csv_count` is the number of CSV files currently in that bucket after cleanup. `meaning` explains what belongs there; `action` records what changed.

| bucket | csv_count | meaning | action |
|---|---:|---|---|
| `polymarket/research/data/analysis/csv_outputs/dali/` | 79 | dali / Polymarket research-lineage result tables: A/P blocks, capture diagnostics, sign convention, TFI, and executable-taker tests. | Moved from flat `data/analysis/*.csv`. |
| `polymarket/research/data/analysis/csv_outputs/market_making/` | 14 | MM result tables: K1/K2/K5/K-PEG maker economics, quoting, stress, and real-maker diagnostics. | Moved from flat `data/analysis/*.csv`. |
| `polymarket/research/data/analysis/csv_outputs/options_delta/` | 14 | OD result tables: K3/K4/K6/K7 lead-lag, basis, vol, arb, and lifecycle diagnostics. | Moved from flat `data/analysis/*.csv`. |
| `polymarket/research/data/analysis/csv_outputs/copytrade/` | 2 | Copytrade bias/cohort diagnostic tables. | Moved from flat `data/analysis/*.csv`. |
| `polymarket/research/data/backtests/paper_journals/` | 2 | Old paper-trading journal CSVs from dali CLOB AI product tests. These are event journals, not analysis summary tables. | Moved out of the backtests root into a journal-specific subfolder. |

## Read Of The Cleanup

Before the cleanup, `polymarket/research/data/analysis/` held 109 CSV files directly at the top level. After the cleanup, the top level has zero CSV files and all 109 analysis CSVs live under `data/analysis/csv_outputs/<cluster>/`. The two paper-journal CSVs bring the project CSV audit total to 111, excluding `.venv`.

The two `paper_journal` CSVs are kept because they are real historical artifacts, but they now live in `data/backtests/paper_journals/` so they do not look like core backtest parquet/json outputs.

## Convention Going Forward

- New human-readable report/result CSVs go under `polymarket/research/data/analysis/csv_outputs/<cluster>/`.
- Use one of the existing cluster folders when possible: `copytrade`, `dali`, `market_making`, or `options_delta`.
- Markdown findings must link or name the exact CSV path they interpret, and any non-obvious table columns must be explained in the note before or after the table.
- Large canonical panels, ledgers, and append-only datasets should remain parquet unless there is a specific reason to emit a small CSV summary.
- Avoid new flat files directly under `polymarket/research/data/analysis/`.

## Future Placeholder Caveat

The external OFI/TOB/L2 research note names future placeholder A2 outputs under `data/analysis/csv_outputs/dali/`. Those files do not exist today, but the placeholders now point at the organized CSV layout so future implementation does not recreate flat `data/analysis/*.csv` files.
