"""Compression pipeline: finds completed JSONL.gz capture files, parses them
into typed Parquet tables (one per event type), validates row counts, and stages
the output for cloud sync."""
