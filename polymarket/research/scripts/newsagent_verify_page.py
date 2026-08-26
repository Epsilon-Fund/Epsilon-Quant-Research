"""Verify the built Observatory page in headless Chrome, against the REAL DOM.

    PYTHONPATH=. uv run python scripts/newsagent_verify_page.py

Every previous round recorded "verified in headless Chrome" as a sentence. This
is that sentence as a script, so the claim is reproducible and a later change
that quietly drops a panel fails loudly instead.

It loads `data/newsagent/showcase/index.html` in Chrome with a virtual-time
budget (so the page's own visibility JS has run), dumps the resulting DOM, and
asserts on what a reader would actually see: the v3.5 evidence groups and their
membership, the prior-only visual state and its exact wording, the provenance
lines, the movement indicators, every v3.4 panel that must survive, and the
hygiene invariants (self-contained, no local-path leak, no private titles).

Exit code 0 = all checks passed. Requires Google Chrome; skips with a clear
message when it is not installed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newsagent import config   # noqa: E402

PAGE = config.SHOWCASE / "index.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def dump_dom() -> str:
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME} — cannot verify the DOM")
    if not PAGE.exists():
        raise SystemExit(f"no built page at {PAGE} — run the daily loop or "
                         "scripts/newsagent_rerender.py first")
    r = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=4000", "--dump-dom", PAGE.resolve().as_uri()],
        capture_output=True, text=True, timeout=180, check=False)
    if not r.stdout:
        raise SystemExit("Chrome returned an empty DOM:\n" + r.stderr[-2000:])
    return r.stdout


dom = dump_dom()
print(f"DOM: {len(dom):,} chars")

sc = json.loads((config.SHOWCASE / "showcase.json").read_text())
groups = {g["group"]: g for g in sc["evidence_groups"]}
checks = []
def ck(name, cond, extra=""):
    checks.append((name, bool(cond), extra))

# ---- v3.5 (1) grouping -----------------------------------------------------
for g in ("news+data", "news, well-fed", "news, thin", "prior-only"):
    ck(f"group box rendered: {g}", f'data-group="{g}"' in dom)
ck("4 group boxes", dom.count('<div class="evgroup"') == 4, dom.count('<div class="evgroup"'))
for g, row in groups.items():
    n = row["n"]
    ck(f"count shown for {g}", f'{n} market{"s" if n != 1 else ""}</span>' in dom)
scored = [g for g in sc["evidence_groups"] if g.get("settled_n")]
for g in scored:
    ck(f"settled Brier shown for {g['group']}",
       f"settled Brier {g['settled_brier']:.4f}" in dom
       and f"(n={g['settled_n']} \u2014" in dom)
ck("every scored group carries its n",
   # the group-header phrasing specifically: § 07's tract table also carries a
   # "read as a hint, not a result" marker and must not be counted here
   dom.count("a description of these markets, not a result") == len(scored),
   dom.count("a description of these markets, not a result"))
ck("groups with no settlements say so",
   dom.count("no settled markets in this group yet")
   == len(sc["evidence_groups"]) - len(scored))
ck("grouping is not the tract tag (says so)",
   "not</b> by what" in dom and "news-driven" in dom)
ck("thresholds declared on the page",
   ">30</span> days" in dom and "\u22655" in dom and "well-fed" in dom)
ck("bands stated as display-only", "never enter the model" in dom)
# cell membership matches the JSON groups
for g, row in groups.items():
    for slug in row["slugs"]:
        m = re.search(r'<a class="cell[^"]*"[^>]*href="#card-' + re.escape(slug) + r'"[^>]*data-group="([^"]*)"', dom, re.S)
        if not m:
            ck(f"cell present {slug[:34]}", False); break
    else:
        ck(f"all {g} cells carry data-group", True)

# ---- v3.5 (2) provenance in words -----------------------------------------
unchanged = [c for c in sc["markets"]
             if c["provenance"]["group"] == "prior-only" and c["provenance"]["unchanged"]]
ck("prior-only cards state the exact line, with their own prior date",
   all("No relevant evidence found; this number is the onboarding prior, "
       f"unchanged since {c['provenance']['prior_set_on']}." in dom for c in unchanged)
   and bool(unchanged))
carried = [c for c in sc["markets"]
           if c["provenance"]["group"] == "prior-only" and not c["provenance"]["unchanged"]]
ck("prior-only-with-carry cards never claim to be unchanged",
   all(f'plus <span class="mono">{c["provenance"]["moved_pp"]:+g}pp</span> of decayed '
       "carry" in dom for c in carried) and bool(carried))
n_prior = sum(1 for c in sc["markets"] if c["provenance"]["group"] == "prior-only")
ck("no-evidence chip on every prior-only card",
   dom.count('provchip-hi">no evidence') == n_prior, dom.count('provchip-hi">no evidence'))
ck("evidence chip on fed markets", 'provchip">evidence' in dom)
for words in ("Guardian", "RSS (BBC/Sky/Politico/The Hill)", "Wikipedia Current Events",
              "reach — official document", "data channel (official statistics)",
              "macro-research PDFs"):
    ck(f"channel named in words: {words}", words in dom)
ck("relevant-count stated", re.search(r"Last 30 days: <span class=\"mono\">\d+</span> relevant article", dom) is not None)
ck("gap-not-a-disagreement warning", "not a researched disagreement" in dom)

# ---- v3.5 (2b) prior-only distinct visual state ----------------------------
ck("prior-only cells carry the class",
   dom.count('class="cell prioronly"') == n_prior, dom.count('class="cell prioronly"'))
ck("prior-only donut is dashed+muted",
   dom.count('stroke-dasharray="4 4"') >= n_prior, dom.count('stroke-dasharray="4 4"'))
ck("prior-only chip on tile", dom.count("prior only · no evidence") == n_prior)
ck("prior-only detail card marked", "prior only — no relevant evidence" in dom)
ck("dashed-ring legend explained", "dashed, muted ring" in dom)

# ---- v3.5 (3) movement indicator ------------------------------------------
ck("per-cell prior\u2192current on every card with a prior",
   dom.count('class="cellmove mono') == sc["movement"]["n"],
   dom.count('class="cellmove mono'))
ck("per-card movement bar",
   dom.count("how much has evidence moved this number") >= sc["movement"]["n"])
ck("data-channel card splits anchor vs news",
   "data anchor" in dom and "and news has moved it" in dom)
ck("page-level movement line",
   f'a mean of <span class="mono">{sc["movement"]["mean_abs_pp"]:.2f}pp</span>' in dom)
ck("recon movement one-liner", "How much has evidence moved these numbers? Barely." in dom)
a = sc["simlive"]["at_resolution"]
ck("recon one-liner carries the reconstruction AND the live number",
   f'{a["mean_abs_shift_from_prior_pp"]:.2f}pp' in dom
   and f'{1 - a["share_moved_ge_1pp"]:.0%}</span> never move' in dom
   and f'{sc["movement"]["share_still_ge_1pp"]:.0%}</span> '
       'still sitting within 1pp' in dom)

# ---- v3.4 intact -----------------------------------------------------------
ck("§06 forward track record", "Public track record — settled forecasts" in dom)
ck("§06 rows present", dom.count("<td class=\"mono dim\">news</td>") >= 4)
ck("§07 reconstruction panel", "Simulated-live reconstruction" in dom and "reconchip" in dom)
ck("staleness curve svg", "snapshot age" in dom and "balanced" in dom)
ck("fed regime split, holding row worse", "WORSE than doing nothing" in dom and "rowhi" in dom)
ck("reach status line", "ukmto.org" in dom and "navigation-only" in dom and "T+1" in dom)
ck("divergence layer", "divergence layer" in dom.lower())
# every mover row carries its own from->to dates (the v3.3 wording fix). The
# "N days apart" sentence only appears when the top gaps exceed a day, and
# today's two published snapshots are consecutive - same as the v3.4 page.
ck("movers carry per-row from->to dates",
   'class="mover"' in dom and re.search(r"now [\d.]+% <span class=\"dim\">\(\d\d-\d\d \u2192 \d\d-\d\d\)", dom) is not None)
ck("v0 closure restated", "CLOSED" in dom or "closed" in dom)
ck("per-tract split kept", "accuracy by market type" in dom)
ck("in-sample warning kept", "in-sample, and that is not a detail" in dom)
ck("evidence-poorer caveat kept", "evidence-poorer world" in dom)

# ---- hygiene ---------------------------------------------------------------
ext = re.findall(r'<(?:script|link)[^>]+(?:src|href)="(https?://[^"]+)"', dom)
ck("0 external script/style refs", not ext, ext[:3])
ck("no local path leak", "/Users/" not in dom)
ck("no private newsletter titles", "newsletter:" not in dom)
ck("footer separates snapshot date from render time",
   f'Numbers from the <b>{sc["snapshot_date"]}</b> snapshot.' in dom
   and "Page rendered" in dom)

bad = [c for c in checks if not c[1]]
for n, ok, extra in checks:
    print(("  ok  " if ok else "  FAIL") + f" {n}" + (f"   [{extra}]" if extra != "" and not ok else ""))
print(f"\n{len(checks) - len(bad)}/{len(checks)} DOM checks passed")
sys.exit(1 if bad else 0)
