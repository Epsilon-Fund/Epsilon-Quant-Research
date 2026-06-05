"""Run copy-execution audits on the 5 leaders requested by the synthesis v2 task.

Each audit reuses scripts/domah_copy_audit.py:main() with leader/label/out_subdir
parameters, so the methodology is identical to the existing Domah and 0xee00ba
audits.

Run from polymarket/research/:
    PYTHONPATH=. python3 scripts/run_leader_audits_batch_v2.py
"""
from __future__ import annotations

import time
from pathlib import Path

# Reuse the existing audit pipeline as a library.
from scripts.domah_copy_audit import main as run_audit
# Augmented FAMILY_KEYWORDS — original + PROPOSED_ADDITIONS from the prior
# validation work (data/analysis/domah_followups/family_heuristic_validation.md
# and used by the ee00ba audit + cross_leader_analysis.py).
from scripts.domah_family_validation import _augment_rules

LEADERS = [
    # (short_label, address, reason)
    ("top_leaderboard",        "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",
     "$14.95M lifetime PnL — does the top of the leaderboard have more deployable cells?"),
    ("high_conviction",        "0x204f72f35326db932158cba6adff0b9a1da95e14",
     "$10.6M PnL, hold_to_resolution_share=74.4% — extreme conviction outlier"),
    ("ultra_maker",            "0x2005d16a84ceefa912d4e380cd32e7ff827875ea",
     "$8.4M PnL, role_balance=0.90 — most maker-heavy candidate; tests adverse-selection generalisation"),
    ("negrisk_directional_1",  "0x629bc4a1e53e1d475beb7ea3d388791e96dd995a",
     "$1.8M PnL, ex-Pool-C NegRisk specialist reclassified two_sided_directional — closes the loop"),
    ("negrisk_directional_2",  "0x5bffcf561bcae83af680ad600cb99f1184d6ffbe",
     "$3.3M PnL, politics-dominant directional NegRisk bettor — generalisation of Domah's family pattern"),
]


def main() -> None:
    t_total = time.time()
    rules = _augment_rules()
    n_kw = sum(len(kws) for _, kws in rules)
    print(f"running {len(LEADERS)} audits sequentially using AUGMENTED FAMILY_KEYWORDS "
          f"({n_kw} keywords across {len(rules)} families)...", flush=True)
    for i, (short_label, address, reason) in enumerate(LEADERS, start=1):
        label = f"leader_{short_label}"
        out_subdir = label  # data/analysis/leader_<short>/<label>_audit_*.parquet
        print(f"\n{'='*72}\n[{i}/{len(LEADERS)}] {label}  ({address})\n  {reason}\n{'='*72}", flush=True)
        t0 = time.time()
        run_audit(leader=address, label=label, out_subdir=out_subdir,
                  family_keywords=rules)
        print(f"\n[{i}/{len(LEADERS)}] {label} done in {time.time()-t0:,.0f}s", flush=True)

    print(f"\nALL DONE in {time.time()-t_total:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
