from __future__ import annotations

from polymarket.execution.mirror.clob_signer import ClobSigner, ClobSignerConfig

NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"


class _OrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Options:
    def __init__(self, *, neg_risk: bool, tick_size: float):
        self.neg_risk = neg_risk
        self.tick_size = tick_size


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_order(self, order_args, options):  # noqa: ARG002
        verifying = NEG_RISK_CTF_EXCHANGE if options.neg_risk else CTF_EXCHANGE
        return {"domain": {"verifyingContract": verifying}}


def _signer() -> ClobSigner:
    signer = ClobSigner(ClobSignerConfig(api_url="https://clob.polymarket.com"))
    signer._clob_client_cls = _Client
    signer._order_args_cls = _OrderArgs
    signer._options_cls = _Options
    return signer


def test_negrisk_true_uses_negrisk_verifying_contract() -> None:
    signed = _signer()({
        "token_id": "asset",
        "price": "0.50",
        "size": "1",
        "side": "BUY",
        "_neg_risk": True,
        "_tick_size": 0.01,
    }, "private-key")
    assert signed["domain"]["verifyingContract"] == NEG_RISK_CTF_EXCHANGE


def test_negrisk_false_uses_binary_verifying_contract() -> None:
    signed = _signer()({
        "token_id": "asset",
        "price": "0.50",
        "size": "1",
        "side": "BUY",
        "_neg_risk": False,
        "_tick_size": 0.01,
    }, "private-key")
    assert signed["domain"]["verifyingContract"] == CTF_EXCHANGE
