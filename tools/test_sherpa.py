#!/usr/bin/env python3
"""
test_sherpa.py — unit tests for the Sherpa skill router's matcher.

Covers, per the semantic-backend swap (fastembed → TF-IDF/BM25, no daemon):
  - keyword layer: deterministic, works standalone
  - graceful degradation: an unavailable backend never hard-fails; keyword survives
  - TF-IDF backend (scikit-learn): active + ranks a lexical query
  - fastembed backend: real model when importable (synonym query surfaces calibrate),
    plus a mocked-backend test of the cache/cosine integration that needs no download

Run:  python3 tools/test_sherpa.py         (stdlib unittest, no pytest needed)
      python3 -m pytest tools/test_sherpa.py
No network, no daemon, no Ollama at any point.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sherpa  # noqa: E402

_HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _skill(name, description, scope="internal"):
    """A synthetic index entry with the same shape build_index() produces."""
    return {
        "name": name,
        "description": description,
        "keywords": sherpa.derive_keywords(name, description, []),
        "scope": scope,
        "repo": "test",
        "source": f".agents/skills/{name}/SKILL.md",
        "global": False,
    }


# A small, hermetic catalog — independent of the real repo's installed skills.
INDEX = [
    _skill("calibrate",
           "Score how well-calibrated a model or forecaster is: Brier score, "
           "log-loss, reliability diagram, recalibrate probabilities (isotonic / Platt), "
           "compare model-p vs market-implied-p."),
    _skill("superforecast",
           "Turn vague concerns and decisions into resolvable probabilistic "
           "predictions. Use when the user asks will X happen, how likely is Z, "
           "or wants calibrated probability tracking."),
    _skill("changepoint-audit",
           "Detect structural breaks / regime shifts in a timestamped series, "
           "lookahead-free, using CUSUM / Page-Hinkley / BOCPD."),
    _skill("prd-scaffold",
           "Co-author a PRD through structured Q&A, then emit one self-contained "
           "goal prompt an implementation agent can execute.", scope="shareable"),
]


class KeywordLayer(unittest.TestCase):
    def test_deterministic(self):
        a, _ = sherpa.rank("regime shift structural break detection", INDEX,
                           use_semantic=False)
        b, _ = sherpa.rank("regime shift structural break detection", INDEX,
                           use_semantic=False)
        self.assertEqual([e["name"] for e in a], [e["name"] for e in b])

    def test_keyword_only_backend_is_none(self):
        top, backend = sherpa.rank("detect a regime shift", INDEX, use_semantic=False)
        self.assertEqual(backend, "none")
        self.assertEqual(top[0]["name"], "changepoint-audit")

    def test_lexical_query_surfaces_expected(self):
        top, _ = sherpa.rank("reliability diagram brier score calibration", INDEX,
                             use_semantic=False)
        self.assertEqual(top[0]["name"], "calibrate")


class GracefulDegradation(unittest.TestCase):
    def test_unknown_backend_falls_to_none(self):
        scores, backend = sherpa.semantic_scores("anything", INDEX,
                                                 backends=["does-not-exist"])
        self.assertEqual((scores, backend), ({}, "none"))

    def test_missing_backend_never_raises_and_keyword_survives(self):
        # Force a backend list of only unavailable names → semantic yields nothing,
        # but rank() must still return keyword-ranked results, never crash.
        top, backend = sherpa.rank("reliability diagram calibration", INDEX,
                                   backends=["nope1", "nope2"])
        self.assertEqual(backend, "none")
        self.assertEqual(top[0]["name"], "calibrate")

    def test_bm25_import_error_is_swallowed(self):
        # rank-bm25 is not a hard dependency; requesting it when absent must be safe.
        if importlib.util.find_spec("rank_bm25") is None:
            scores, backend = sherpa.semantic_scores("x", INDEX, backends=["bm25"])
            self.assertEqual(backend, "none")


@unittest.skipUnless(_HAS_SKLEARN, "scikit-learn not installed")
class TfidfBackend(unittest.TestCase):
    def test_active_and_ranks(self):
        scores, backend = sherpa.semantic_scores(
            "brier score reliability calibration", INDEX, backends=["tfidf"])
        self.assertEqual(backend, "tfidf")
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))
        # lexical overlap makes calibrate the top TF-IDF hit
        self.assertEqual(max(scores, key=scores.get), "calibrate")

    def test_rank_uses_tfidf(self):
        top, backend = sherpa.rank("brier score reliability calibration", INDEX,
                                   backends=["tfidf"])
        self.assertEqual(backend, "tfidf")
        self.assertEqual(top[0]["name"], "calibrate")


class FastembedMockedIntegration(unittest.TestCase):
    """Exercise the _semantic_fastembed cache/cosine path with a FAKE fastembed
    module — verifies our integration code (cache keying, reuse, cosine) with no
    model download, so it runs anywhere."""

    def setUp(self):
        import types
        # deterministic 3-dim "embeddings" from keyword presence
        def _vec(text):
            t = text.lower()
            return [float(t.count("probabil") + t.count("calibrat")),
                    float(t.count("regime") + t.count("break")),
                    float(t.count("prd") + t.count("plan"))]

        class FakeTextEmbedding:
            def __init__(self, model_name=None):
                pass
            def embed(self, texts):
                for t in texts:
                    yield _vec(t)

        self.fake = types.ModuleType("fastembed")
        self.fake.TextEmbedding = FakeTextEmbedding
        self._saved = sys.modules.get("fastembed")
        sys.modules["fastembed"] = self.fake

    def tearDown(self):
        if self._saved is not None:
            sys.modules["fastembed"] = self._saved
        else:
            sys.modules.pop("fastembed", None)

    def test_scores_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "emb.json"
            scores, backend = sherpa.semantic_scores(
                "how good are my probability estimates", INDEX,
                cache_path=cache, backends=["fastembed"])
            self.assertEqual(backend, "fastembed")
            # query aligns with the probability axis → calibrate & superforecast on top
            self.assertGreater(scores["calibrate"], scores["changepoint-audit"])
            self.assertGreater(scores["superforecast"], scores["prd-scaffold"])
            # cache written and reusable
            self.assertTrue(cache.is_file())
            cached = json.loads(cache.read_text())
            self.assertTrue(any(k.startswith("fastembed:") for k in cached))
            # second call reuses the cache (no error, same result)
            scores2, _ = sherpa.semantic_scores(
                "how good are my probability estimates", INDEX,
                cache_path=cache, backends=["fastembed"])
            self.assertEqual(scores, scores2)


@unittest.skipUnless(_HAS_FASTEMBED, "fastembed not installed")
class FastembedRealModel(unittest.TestCase):
    """The real ONNX model (downloaded once). No daemon; fully in-process."""

    def test_synonym_query_surfaces_calibrate(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "emb.json"
            top, backend = sherpa.rank(
                "how good are my probability estimates", INDEX, top_n=3,
                cache_path=cache, backends=["fastembed"])
            self.assertEqual(backend, "fastembed")
            names = [e["name"] for e in top]
            self.assertIn("calibrate", names,
                          f"semantic layer should surface calibrate; got {names}")

    def test_semantic_catches_what_keyword_misses(self):
        # Zero lexical overlap: the query shares no token with calibrate's text,
        # so keyword-only does NOT surface it — but fastembed does. This is the
        # whole point of the semantic layer, and it needs no running daemon.
        q = "how good are my probability estimates"
        kw_top, kw_backend = sherpa.rank(q, INDEX, top_n=5, use_semantic=False)
        self.assertEqual(kw_backend, "none")
        self.assertNotIn("calibrate", [e["name"] for e in kw_top])
        with tempfile.TemporaryDirectory() as d:
            se_top, se_backend = sherpa.rank(q, INDEX, top_n=3,
                                             cache_path=Path(d) / "e.json",
                                             backends=["fastembed"])
            self.assertEqual(se_backend, "fastembed")
            self.assertIn("calibrate", [e["name"] for e in se_top])


if __name__ == "__main__":
    unittest.main(verbosity=2)
