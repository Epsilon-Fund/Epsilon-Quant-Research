"""Operator/MM-bot deny-list, sourced from the validation report.

Every address listed here is excluded from `traders_filtered` by default.
The cohort-selection layer (Phase 4+) reads `traders_filtered`; raw
metric data for these addresses still lives in `traders_raw` for analysis.

Categories follow the validation report's clustering:
  - Cluster A (PURE_RELAYERS):   matcher/relayer; 0 maker fills, ~2M counterparties
  - Cluster B (PURE_MM_BOTS):    pure liquidity providers; maker:taker > 50
  - Cluster C (HFT):             matched arb flow; ratio ~1.0, >95% sub-second clustering
"""

PURE_RELAYERS = frozenset({
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",  # 305M fills, $17.3B
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",  # 100M fills, $12.6B
})

PURE_MM_BOTS = frozenset({
    "0x297fbd45782af37d899015aebbc52437f3d55103",  # ratio 870k:1
    "0x04895657d3c2afebec8be4b6e60b9c56ad68ee4d",  # ratio 1116:1
    "0xdc669ba0adb45448020025f756070492d1070533",  # ratio 687:1
    "0xe9cbb1c9b3f7f411dd4fdf2ea7afa780c8b4d096",  # ratio 78:1
    "0x38e598961dd0456a7fb2e758bd433d3e59fb8a4a",  # ratio 45:1
    "0xd44e29936409019f93993de8bd603ef6cb1bb15e",  # ratio 12:1
    "0x5f4d4927ea3ca72c9735f56778cfbb046c186be0",  # ratio 13:1
})

HFT = frozenset({
    "0xe8dd7741ccb12350957ec71e9ee332e0d1e6ec86",  # ratio 0.99, 99% sub-sec
    "0x63d43bbb87f85af03b8f2f9e2fad7b54334fa2f1",  # ratio 1.02, 98% sub-sec
    "0xe3726a1b9c6ba2f06585d1c9e01d00afaedaeb38",  # ratio 1.07, 97% sub-sec
})

OPERATOR_ADDRESSES: frozenset[str] = PURE_RELAYERS | PURE_MM_BOTS | HFT


def is_operator_like(
    maker_taker_ratio: float | None,
    n_fills: int | None,
    distinct_counterparties: int | None,
    pct_sub_second: float | None,
) -> bool:
    """Operator-detection heuristic from the validation report.

    Returns True if the trader's profile matches any of:
      - extreme maker:taker ratio (> 50 or < 0.02)
      - very large counterparty fan-out (> 500k)
      - high sub-second clustering (> 95%) AND high fill count (> 1M)

    NULLs are treated as "no signal" — they don't trigger.
    """
    if maker_taker_ratio is not None:
        if maker_taker_ratio > 50.0 or maker_taker_ratio < 0.02:
            return True
    if distinct_counterparties is not None and distinct_counterparties > 500_000:
        return True
    if (
        pct_sub_second is not None and pct_sub_second > 95.0
        and n_fills is not None and n_fills > 1_000_000
    ):
        return True
    return False
