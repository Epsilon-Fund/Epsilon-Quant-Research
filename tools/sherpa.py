#!/usr/bin/env python3
"""
sherpa.py — local skill router (auto-surface skills on task context).

Given a task/prompt, score every skill installed in this repo and return the
top-N most relevant, each with a one-line "use when". This is the proactive
surfacing layer on top of Claude Code's native description-triggering: the
bootstrap step in the brain law files runs this at session start / when the
task shifts, so the right skills load without anyone naming them.

Matcher = keyword/description scoring (always on, deterministic, offline) blended
with LOCAL semantic similarity (optional booster). The semantic layer runs
IN-PROCESS with no daemon: it prefers `fastembed` (in-process ONNX sentence
embeddings — true synonym matching), and if that isn't importable it falls back
to TF-IDF cosine (`scikit-learn`) and then BM25 (`rank-bm25`), both pure-Python
and model-free. Nothing ever leaves the machine (fastembed downloads its small
model once, then runs fully offline). If no semantic backend is available, Sherpa
degrades cleanly to keyword-only; the ranking stays sane and deterministic.

Self-contained: stdlib only at the core; the semantic backends are OPTIONAL,
guarded imports (missing ones are skipped, never a hard failure). No cloud, no
dependency on skills_catalog.py — it scans SKILL.md frontmatter directly, so the
same file drops into any repo that has a skills directory.

Usage:
  python3 tools/sherpa.py "run a CPCV sweep over the momentum assets"
  python3 tools/sherpa.py --top 3 --json "how well calibrated is my model?"
  python3 tools/sherpa.py --scope shareable "help me spec this feature"
  python3 tools/sherpa.py --reindex-embeddings "..."   # force re-embed (fastembed)
  python3 tools/sherpa.py --list                        # dump the index

Exit code is always 0 on a successful run (even with zero matches).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

# ── repo / config discovery ──────────────────────────────────────────────────

# Markers that identify a repo root when walking up from CWD or this file.
_ROOT_MARKERS = (".git", "brain", "tools", ".agents")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file's dir) to the nearest repo root.
    A root is any dir containing one of _ROOT_MARKERS. Falls back to CWD."""
    start = (start or Path(__file__).resolve().parent)
    for cand in (start, *start.parents):
        if any((cand / m).exists() for m in _ROOT_MARKERS):
            # prefer a dir that actually holds a skills location
            if any((cand / d).exists() for d in (".agents/skills", ".claude/skills",
                                                 "library/skills", "tools")):
                return cand
    return Path.cwd()


def skill_dir_globs(root: Path, include_global: bool = False) -> list[tuple[Path, str]]:
    """(glob-root, glob-pattern) pairs for every place a SKILL.md can live.

    REPO-LOCAL ONLY by default — this is the cross-repo privacy guarantee: a
    machine-global skills dir (~/.codex/skills, ~/.claude/skills) may hold ANOTHER
    repo's skills (e.g. epsilon symlinks its internal skills into ~/.codex/skills),
    so scanning it would surface one repo's internals inside another. Sherpa
    routes the CURRENT repo's skills. Pass include_global=True to additionally
    surface machine-global harness skills (query, systematic-debugging, …) — opt-in,
    and only sensible in the repo that owns those globals."""
    specs = [
        (root / ".agents" / "skills", "*/SKILL.md"),
        (root / ".claude" / "skills", "*/SKILL.md"),   # repo-local (vendored copies / symlinks)
        (root / "library" / "skills", "*/SKILL.md"),
        (root / "library", "*/src/**/skills/*/SKILL.md"),
        (root / "skills", "*/SKILL.md"),               # cross-repo vendored location
    ]
    if include_global:
        home = Path.home()
        specs += [
            (home / ".claude" / "skills", "*/SKILL.md"),
            (home / ".codex" / "skills", "*/SKILL.md"),
        ]
    return [(base, pat) for base, pat in specs if base.exists()]


# ── scope tagging (internal vs shareable) ────────────────────────────────────
# Curated map wins; unknown skills fall to a heuristic. An optional
# tools/sherpa_scope.json in the repo overrides/extends this map, so each repo
# can tag its own local skills without editing this file.
#
# NOTE on calibrate / changepoint-audit: the `.agents/skills/` INSTALLS are
# epsilon-wired (they call the epsilon ledger / infrastructure CLIs) and tag
# `internal` via the body heuristic below; only their scrubbed `library/` BUNDLE
# forms are `shareable`. That split is intentional — the cross-repo copy vendors
# the library form, never the .agents install. So neither name is pinned here.
_DEFAULT_SCOPE = {
    # shareable — scrubbed / generic, no epsilon internals, safe to vendor out.
    # (These live in .agents/skills or library/skills and carry no epsilon-path
    #  hints, so the body heuristic can't infer them; pin them explicitly.)
    "prd-scaffold": "shareable",
    "reflection-prompt": "shareable",
    "cost-mode": "shareable",
    "audio-transcribe-summarize": "shareable",
    # internal — epsilon-wired, must never leave the repo. (Most also trip the
    #  body heuristic; pinned as a belt-and-suspenders fallback.)
    "data-contract": "internal",
    "superforecast": "internal",
    "superforecasting": "internal",
    "efficient-fable": "internal",
    "stay-within-limits": "internal",
    "reflection-engine": "internal",
    "brain-cartographer": "internal",
    "brain-chronicler": "internal",
    "brain-janitor": "internal",
    "brain-librarian": "internal",
    "brain-rock-tumbler": "internal",
    "find-skills": "internal",           # Sherpa itself stays internal this pass
}

# When the SAME skill name exists both epsilon-wired (.agents/skills) and as a
# scrubbed library bundle, the source path disambiguates the tag.
_EPSILON_HINTS = re.compile(
    r"\b(epsilon|polymarket|/research/|SF_BOOK|infrastructure\.|topics/|live_trading|"
    r"brain/reflection|brain/generated|data_infra)\b", re.IGNORECASE)


def load_scope_overrides(root: Path) -> dict:
    f = root / "tools" / "sherpa_scope.json"
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def derive_scope(skill_id: str, source: str, body: str, overrides: dict) -> str:
    """Scope of THIS SKILL.md file. Priority:
      1. repo-local sherpa_scope.json override
      2. a library/ bundle path ⇒ scrubbed public candidate ⇒ shareable
      3. curated pin (authoritative for named skills — set before the body
         heuristic so a generic skill isn't mis-tagged internal just because its
         provenance comment mentions 'epsilon-quant-research')
      4. epsilon internals in the body ⇒ internal (catches the .agents installs
         of calibrate/changepoint-audit, which are deliberately NOT pinned so a
         shareable library form and an internal install can coexist)
      5. unknown ⇒ leave for human review
    """
    if skill_id in overrides:
        return overrides[skill_id]
    src = source.replace(os.sep, "/")
    if "/library/" in src or src.startswith("library/"):
        return "shareable"
    if skill_id in _DEFAULT_SCOPE:
        return _DEFAULT_SCOPE[skill_id]
    if _EPSILON_HINTS.search(body or ""):
        return "internal"
    return "unknown"


# ── SKILL.md frontmatter ─────────────────────────────────────────────────────
def parse_frontmatter(text: str, fallback_name: str) -> dict:
    """name + description (+ optional keywords) from YAML frontmatter.
    Handles `key: >`/`|` folded/literal blocks. No YAML dependency."""
    out: dict = {"name": fallback_name, "description": "", "keywords": []}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return out
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        km = re.match(r"^([\w][\w-]*):\s*(.*)$", line)
        if km:
            key, val = km.group(1), km.group(2).strip()
            if val in (">", "|", ">-", "|-", ">+", "|+"):
                block = []
                i += 1
                while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                    block.append(lines[i].strip())
                    i += 1
                out[key] = " ".join(b for b in block if b)
                continue
            out[key] = val
        i += 1
    # normalize keywords into a list
    kw = out.get("keywords") or out.get("trigger_keywords") or []
    if isinstance(kw, str):
        kw = [k.strip() for k in re.split(r"[,;]", kw) if k.strip()]
    out["keywords"] = kw
    return out


# ── tokenization + keyword scoring ───────────────────────────────────────────
_STOP = frozenset("""
a an the this that these those and or but if then else when while for of to in on
at by with from into over under is are was were be been being it its it's as no not
you your we our i me my they them their he she his her use used using when whenever
run runs running want wants need needs help do does doing done make makes made get
gets got give gives one two per via across over etc via so than not never always
every each any some all more most less least new old good bad how what why where
which who whom whose about after before again once here there also just only very
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\_/]*")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        t = t.strip("-._/")
        if len(t) < 3 or t in _STOP:
            continue
        toks.append(t)
    return toks


def derive_keywords(name: str, description: str, explicit: list[str]) -> list[str]:
    """Salient trigger keywords for a skill: explicit frontmatter keywords, the
    name parts, then the most frequent non-stopword description tokens."""
    seen: dict[str, int] = {}
    order: list[str] = []
    def add(tok: str):
        if tok and tok not in seen:
            seen[tok] = 1
            order.append(tok)
    for k in explicit:
        for t in tokenize(k):
            add(t)
    for t in name.replace("-", " ").replace("_", " ").split():
        add(t.lower())
    # frequency-ranked description tokens (deterministic: freq desc, then alpha)
    freq: dict[str, int] = {}
    for t in tokenize(description):
        freq[t] = freq.get(t, 0) + 1
    for t in sorted(freq, key=lambda x: (-freq[x], x)):
        add(t)
    return order[:24]


def keyword_score(task_tokens: list[str], skill: dict) -> float:
    """Overlap of task tokens with a skill's weighted term bag. Deterministic.
    Weights: explicit keywords/name 3x, description terms 1x. Normalized so a
    perfectly-covered task tops out near 1.0."""
    if not task_tokens:
        return 0.0
    weights: dict[str, float] = {}
    for kw in skill["keywords"][:12]:
        weights[kw] = max(weights.get(kw, 0.0), 3.0)
    for t in tokenize(skill["name"].replace("-", " ")):
        weights[t] = max(weights.get(t, 0.0), 3.0)
    for t in tokenize(skill["description"]):
        weights.setdefault(t, 1.0)
    task_set = set(task_tokens)
    hit = sum(w for term, w in weights.items() if term in task_set)
    # normalize by the achievable mass from the task side (cap so short tasks
    # can still score high), plus a small idf-ish bonus for rarer term hits.
    denom = 3.0 * min(len(task_set), 8) + 4.0
    return min(hit / denom, 1.0)


# ── semantic layer (in-process, no daemon) ───────────────────────────────────
# Backends tried in order; the first importable one that yields scores wins:
#   1. fastembed — in-process ONNX sentence embeddings (true synonym matching);
#      downloads its small model once, then fully offline. No server.
#   2. tfidf     — scikit-learn TF-IDF cosine (lexical, deterministic, no model).
#   3. bm25      — rank-bm25 Okapi (lexical, deterministic, no model).
# Each is a guarded import: a missing/erroring backend is skipped, never fatal.
# If none are available, semantic is skipped and Sherpa is keyword-only. The
# keyword layer is never affected by any of this.
_FASTEMBED_MODEL = os.environ.get("SHERPA_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
_BACKEND_ORDER = [b.strip() for b in
                  os.environ.get("SHERPA_SEMANTIC_BACKENDS", "fastembed,tfidf,bm25").split(",")
                  if b.strip()]


def _cosine(a, b) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _desc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path) -> dict:
    if cache_path and Path(cache_path).is_file():
        try:
            return json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache_path, cache: dict) -> None:
    if not cache_path:
        return
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def _semantic_fastembed(task, index, cache_path=None, reindex=False) -> dict:
    """In-process ONNX embeddings. Description vectors are cached to disk keyed
    by (model, sha256(description)) — only re-embedded when a description changes;
    the query is embedded fresh each call. Raises ImportError if fastembed absent."""
    from fastembed import TextEmbedding      # optional dep — guarded by caller

    ns = f"fastembed:{_FASTEMBED_MODEL}:"
    cache = {} if reindex else _load_cache(cache_path)
    missing = [s for s in index if (ns + _desc_hash(s["description"])) not in cache]
    # One model instance embeds the query + any uncached descriptions together.
    model = TextEmbedding(model_name=_FASTEMBED_MODEL)
    to_embed = [task] + [s["description"] for s in missing]
    vecs = [list(map(float, v)) for v in model.embed(to_embed)]
    qv, dvecs = vecs[0], vecs[1:]
    for s, v in zip(missing, dvecs):
        cache[ns + _desc_hash(s["description"])] = v
    if missing:
        _save_cache(cache_path, cache)
    return {s["name"]: max(0.0, _cosine(qv, cache[ns + _desc_hash(s["description"])]))
            for s in index}


def _semantic_tfidf(task, index, cache_path=None, reindex=False) -> dict:
    """TF-IDF cosine over descriptions. Deterministic, no model download."""
    from sklearn.feature_extraction.text import TfidfVectorizer   # optional dep
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [s["description"] for s in index]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(docs + [task])
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    return {s["name"]: float(max(0.0, sims[i])) for i, s in enumerate(index)}


def _semantic_bm25(task, index, cache_path=None, reindex=False) -> dict:
    """BM25 Okapi over descriptions, normalized to [0,1]. Deterministic, no model."""
    from rank_bm25 import BM25Okapi          # optional dep

    corpus = [tokenize(s["description"]) for s in index]
    bm = BM25Okapi(corpus)
    raw = list(bm.get_scores(tokenize(task)))
    top = max(raw) if raw else 0.0
    if top <= 0:
        return {s["name"]: 0.0 for s in index}
    return {s["name"]: float(max(0.0, raw[i]) / top) for i, s in enumerate(index)}


_SEMANTIC_BACKENDS = {
    "fastembed": _semantic_fastembed,
    "tfidf": _semantic_tfidf,
    "bm25": _semantic_bm25,
}


def semantic_scores(task: str, index: list[dict], cache_path=None,
                    reindex: bool = False, backends: list[str] | None = None
                    ) -> tuple[dict[str, float], str]:
    """Return (scores_by_skill, backend_name). Tries backends in order; the first
    that imports and returns scores wins. ({}, 'none') if none are available —
    the caller then routes on keyword score alone. Never raises."""
    if not index:
        return {}, "none"
    for name in (backends or _BACKEND_ORDER):
        fn = _SEMANTIC_BACKENDS.get(name)
        if fn is None:
            continue
        try:
            scores = fn(task, index, cache_path=cache_path, reindex=reindex)
        except Exception:
            continue                                # missing dep / runtime error → next
        if scores:
            return scores, name
    return {}, "none"


# ── index build ──────────────────────────────────────────────────────────────
def build_index(root: Path, include_global: bool = False) -> list[dict]:
    overrides = load_scope_overrides(root)
    by_name: dict[str, dict] = {}
    for base, pattern in skill_dir_globs(root, include_global=include_global):
        for skill_md in sorted(base.glob(pattern)):
            try:
                text = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            meta = parse_frontmatter(text, skill_md.parent.name)
            name = meta["name"] or skill_md.parent.name
            if not meta["description"]:
                continue
            try:
                source = str(skill_md.relative_to(root))
            except ValueError:
                source = str(skill_md)          # global skill outside the repo
            is_global = str(base).startswith(str(Path.home()))
            entry = {
                "name": name,
                "description": meta["description"],
                "keywords": derive_keywords(name, meta["description"], meta["keywords"]),
                "scope": derive_scope(name, source, text, overrides),
                "repo": root.name,
                "source": source,
                "global": is_global,
            }
            # repo-local definitions win over user-global ones of the same name
            existing = by_name.get(name)
            if existing is None or (existing["global"] and not is_global):
                by_name[name] = entry
    return sorted(by_name.values(), key=lambda e: e["name"])


def one_line_use_when(description: str) -> str:
    """The 'Use …' sentence (the surfacing hint), else the first sentence.
    Guards against mid-sentence abbreviations ('e.g.', 'i.e.', 'etc.') and only
    matches a capitalized 'Use' at a sentence boundary (not a lowercase 'use'
    buried in a quote)."""
    d = (description or "")
    for abbr, repl in (("e.g.", "eg"), ("i.e.", "ie"), ("etc.", "etc"),
                       ("vs.", "vs"), ("Dr.", "Dr")):
        d = d.replace(abbr, repl)
    m = re.search(r"(?:^|[.!?]\s+)(Use\b[^.!?]*[.!?])", d)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    first = re.split(r"(?<=[.!?])\s", d.strip(), maxsplit=1)[0]
    return re.sub(r"\s+", " ", first).strip()


# ── ranking ──────────────────────────────────────────────────────────────────
def rank(task: str, index: list[dict], top_n: int = 5,
         scope: str | None = None, use_semantic: bool = True,
         cache_path: Path | None = None, reindex: bool = False,
         backends: list[str] | None = None) -> tuple[list[dict], str]:
    """Return (ranked_top_n, semantic_backend). backend is 'none' when semantic
    is off/unavailable (⇒ keyword-only). Deterministic: ties break on name."""
    pool = [s for s in index if scope is None or s["scope"] == scope]
    task_tokens = tokenize(task)
    kw = {s["name"]: keyword_score(task_tokens, s) for s in pool}

    sem: dict[str, float] = {}
    backend = "none"
    if use_semantic:
        sem, backend = semantic_scores(task, pool, cache_path,
                                       reindex=reindex, backends=backends)
    semantic_used = backend != "none"

    scored = []
    for s in pool:
        k = kw[s["name"]]
        if semantic_used:
            combined = 0.6 * k + 0.4 * sem.get(s["name"], 0.0)
        else:
            combined = k
        scored.append({**s, "score": round(combined, 4),
                       "keyword_score": round(k, 4),
                       "semantic_score": round(sem.get(s["name"], 0.0), 4) if semantic_used else None,
                       "use_when": one_line_use_when(s["description"])})
    scored.sort(key=lambda e: (-e["score"], e["name"]))
    top = [e for e in scored if e["score"] > 0][:top_n]
    return top, backend


# ── CLI ──────────────────────────────────────────────────────────────────────
def _default_cache(root: Path) -> Path:
    gen = root / "brain" / "generated"
    if gen.is_dir():
        return gen / "sherpa_embeddings.json"
    return root / ".sherpa" / "embeddings.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Local skill router — surface the top-N skills for a task.")
    ap.add_argument("task", nargs="*", help="task / prompt text to match against")
    ap.add_argument("--top", type=int, default=5, help="how many skills to surface")
    ap.add_argument("--scope", choices=["internal", "shareable", "unknown"],
                    help="restrict to one scope")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--list", action="store_true", help="dump the full index and exit")
    ap.add_argument("--no-semantic", action="store_true",
                    help="keyword-only (skip the semantic backend)")
    ap.add_argument("--backends", help="comma-separated semantic backend order "
                    "(default: fastembed,tfidf,bm25)")
    ap.add_argument("--reindex-embeddings", action="store_true",
                    help="force re-embed of skill descriptions (fastembed cache)")
    ap.add_argument("--include-global", action="store_true",
                    help="also index machine-global skills (~/.claude, ~/.codex); "
                         "off by default so one repo never surfaces another's skills")
    ap.add_argument("--root", type=Path, help="repo root (default: auto-detect)")
    args = ap.parse_args(argv)

    root = args.root.resolve() if args.root else find_repo_root()
    index = build_index(root, include_global=args.include_global)

    if args.list:
        if args.json:
            print(json.dumps({"repo": root.name, "count": len(index),
                              "skills": index}, indent=2))
        else:
            print(f"# {len(index)} skills indexed in {root.name}")
            for s in index:
                print(f"  [{s['scope']:9}] {s['name']:26} {one_line_use_when(s['description'])[:80]}")
        return 0

    task = " ".join(args.task).strip()
    if not task:
        ap.error("provide a task string, or use --list")

    backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
    top, backend = rank(
        task, index, top_n=args.top, scope=args.scope,
        use_semantic=not args.no_semantic, cache_path=_default_cache(root),
        reindex=args.reindex_embeddings, backends=backends)
    semantic_used = backend != "none"

    if args.json:
        print(json.dumps({
            "repo": root.name, "task": task, "semantic_used": semantic_used,
            "semantic_backend": backend,
            "matcher": f"keyword+semantic ({backend})" if semantic_used else "keyword-only",
            "results": [{k: e[k] for k in ("name", "scope", "score", "keyword_score",
                                           "semantic_score", "use_when", "source")}
                        for e in top],
        }, indent=2))
        return 0

    matcher = (f"keyword+semantic ({backend})" if semantic_used
               else "keyword-only (no semantic backend available)")
    print(f"Sherpa · {root.name} · {matcher}")
    print(f"task: {task}\n")
    if not top:
        print("  (no skill matched — proceed without one, or broaden the task text)")
        return 0
    for i, e in enumerate(top, 1):
        print(f"{i}. {e['name']}  ({e['scope']}, score {e['score']:.2f})")
        print(f"   → {e['use_when']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
