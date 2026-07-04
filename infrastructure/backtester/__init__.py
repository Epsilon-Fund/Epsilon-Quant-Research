"""
infrastructure.backtester — engine, performance/portfolio metrics, ML metrics,
and visualizer for the crypto research stack.

Intentionally re-exports nothing: submodules (engine, performance_metrics,
portfolio_metrics, ml_metrics, visualizer) carry heavy optional deps, so import
them explicitly, e.g. `from infrastructure.backtester import ml_metrics`.
(This file exists so the package is a regular package, not an implicit
namespace package — tooling/pytest resolution was flaky without it.)
"""
