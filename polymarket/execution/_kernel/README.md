# _kernel — frozen vendored execution kernel

Copied from `polymarket/midas/executor/` on **2026-05-06** and **deliberately frozen**: never edited here, never synced with the midas source. The execution module depends on this exact snapshot (venue Protocol, order state machine, idempotency/ambiguous-submit semantics); drift would silently change order-lifecycle behaviour under the tests.

- If a kernel change seems needed: stop and ask — do not edit these files, do not re-vendor without an explicit decision.
- The `# Copied from midas/executor/... on 2026-05-06` headers in each file are provenance comments, not imports; this package is self-contained.
- This is one of **three** CLOB client/signer stacks in the repo, intentionally not deduplicated — see the "CLOB client stacks" section in [../README.md](../README.md) for which stack is source of truth for which consumer.
