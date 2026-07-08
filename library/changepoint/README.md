# lemma-changepoint

Causal, **lookahead-free** structural-break detection for quantitative research:
CUSUM and Page-Hinkley as O(1)/bar live first lines, and Bayesian Online
Changepoint Detection (BOCPD, Adams & MacKay 2007) as the richer second line
with a genuine `change_prob`. Every detector's state at time *t* is a pure
function of data ≤ *t* — appending future bars can never change a past output,
and the test suite asserts exactly that.

## Why this exists

Most changepoint tooling (ruptures, HMM segmentation) is **batch**: it sees the
whole series at once, which makes it lookahead-unsafe for trading research.
This package is the causal counterpart — usable live, bar by bar — plus the
three integrations quant research actually needs:

1. **Purged-CV embargo** — turn break timestamps into purge indices for a
   combinatorial purged cross-validation splitter (`embargo_indices_from_breaks`).
2. **Causal regime features** — `change_prob` / run-length columns safe to feed
   a next-bar regime model (`changepoint_features`).
3. **Fresh-break trend gate** — block trend entries for a cooldown right after a
   break (`fresh_break_gate`).

## Install

No package registry — the repo is the distribution. Two ways in:

**Just the agent skill** (no Python install): copy the bundle into your
agent's skills directory:

```bash
cp -r src/lemma/changepoint/skills/changepoint-audit  .claude/skills/
```

**The engine as a Python package** (deps: numpy + pandas only), straight from
git:

```bash
pip install "lemma-changepoint @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=changepoint"
# extras: [parquet] parquet IO for the CLI/persist · [offline] ruptures (batch labelling ONLY)
```

## Quick start

```python
from lemma.changepoint import run_detector, LiveDetector, breaks_from_stream

stream = run_detector(log_returns, name="bocpd", timestamps=idx)
breaks = breaks_from_stream(stream)          # timestamps where cp_flag fired

live = LiveDetector("cusum", k=0.5, h=5.0)   # real-time hook
row = live.update(ts, x_t)                   # one dict per closed bar
```

CLI:

```bash
lemma-changepoint detect prices.parquet --column Close --returns --standardize \
    --detector bocpd --out changepoints/prices_bocpd.parquet
lemma-changepoint benchmark     # detection lag / false-positive rate
lemma-changepoint kappa-demo    # Cohen's kappa vs Markov-switching transitions
```

## Runnable demo

[`examples/demo.py`](examples/demo.py) is a seeded, self-contained walkthrough —
no data files, no extra deps, a few seconds:

```bash
python examples/demo.py
```

It builds a synthetic calm → crisis → recovery return series with two **known**
breaks, scores all three detectors against that truth (detection lag, misses,
false alarms — including the honest trade-off: CUSUM/PH catch the big break at
lag 0 with zero false alarms but miss the subtle one; BOCPD catches both at the
cost of extra flags), then exercises all three integration helpers and verifies
that a bar-by-bar `LiveDetector` run is identical to the batch run.

## Agent skill bundle

The package ships a Claude-Code-compatible skill bundle
([Agent Skills spec](https://agentskills.io)) and an installer:

```bash
python -m lemma.changepoint.skills install --project   # ./.claude/skills/
python -m lemma.changepoint.skills install --global    # ~/.claude/skills/
```

## Guarantees & benchmarks

- **No lookahead** (tested): the output for bars `1..k` is identical whether or
  not bars `k+1..n` exist.
- **Append-only persistence** (tested): `append_changepoints` never rewrites history.
- Synthetic benchmarks (reproduce with `lemma-changepoint benchmark`):
  mean-shift recall ≈ 1.0 (PH/BOCPD) with 1–4 bar median lag; variance-shift
  recall ≈ 1.0 for BOCPD (mean-only detectors are blind there); false alarms
  < 2 per 1000 stationary bars.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
