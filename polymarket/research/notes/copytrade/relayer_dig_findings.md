---
tags: [dali, block-e-lite, results]
title: Relayer Identification Dig Findings
created: 2026-05-28
status: closed
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
---

# Relayer Identification Dig Findings

> Hub: [[COWORK]]


## Summary

This note identifies the two supposed Block E Lite relayer addresses as Polymarket's legacy CTF Exchange v1 contracts, one standard and one NegRisk. The local fingerprint, Polygonscan labels, Polymarket source/docs, and v2 migration timing all agree. The practical conclusion is that filtering these addresses removes exchange-internal settlement/event-decoding legs, not a retail UI or smart-order-router trader population.

Generated: 2026-05-28

## TL;DR

Critical sanity check: the two "relayer" addresses are not relayers, UI routers, wallet aggregators, or trader infrastructure. They are Polymarket's legacy CTF Exchange v1 contracts: standard CTF Exchange and Neg Risk CTF Exchange. The Block B / Block E Lite lift should therefore be reinterpreted as an exchange-as-intermediary / event-decoding filter, not as evidence that removing retail UI flow or smart batched flow improves the signal.

## 0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e

Classification: **Polymarket-internal contract: legacy CTF Exchange v1**. Confidence: **high**. Polygonscan labels the address as [Polymarket: CTF Exchange](https://polygonscan.com/address/0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e), shows it as a contract with verified source, contract name `CTFExchange`, 0 POL balance, roughly 1.25M transactions, and creator `0x81fd0E5E7372ED171f421A7C33a4b263Ea9DCc25` in the [Sep-26-2022 deployment transaction](https://polygonscan.com/tx/0x35423c49cb07c9ccecad9af20df52cccdeff0d46f833d438de8b02f2504aed22). The Polymarket `ctf-exchange` repo describes the system as a hybrid exchange with offchain matching and onchain settlement, and its deployments table lists Polygon `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` as the CTF Exchange contract ([README](https://github.com/Polymarket/ctf-exchange#deployments)). The source code explains why this address appears in raw `taker`: `_matchOrders` emits an `OrderFilled` event with `taker = address(this)` for the active/taker-order leg ([Trading.sol#L132-L167](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/mixins/Trading.sol#L132-L167)). Polymarket's current V2 migration docs explicitly show this old verifying contract moving from `0x4b...8982E` to V2 `0xE111...996B` ([migration docs](https://docs.polymarket.com/v2-migration)), while current contract docs list the V2 address as canonical now ([contracts docs](https://docs.polymarket.com/resources/contracts)).

Evidence notes:

- Local fingerprint, `trades_seed.parquet`: `as_maker=0`, `as_taker=25,129,291`, first fill `2022-11-21 19:50:09`, last fill `2025-10-07 16:39:50`.
- Local fingerprint, delta shards: `as_maker=0`, `as_taker=288,761,642`, first fill `2025-10-07 16:39:52`, last fill `2026-04-28 11:00:40`.
- The `2026-04-28 11:00:40` local last-seen timestamp lines up with the public CLOB v2 cutover around `2026-04-28 ~11:00 UTC` reported by the community Substreams package ([polymarket-orderbook-substreams](https://github.com/PaulieB14/polymarket-orderbook-substreams#contract-addresses)).

Strategic implication: **If we filter this address, we are excluding standard-market v1 exchange-internal `OrderFilled` legs where the exchange contract is emitted as the taker, which means the filter is a data-model correction for maker/taker attribution, not a trader-population filter.**

## 0xc5d563a36ae78145c45a50134d48a1215220f80a

Classification: **Polymarket-internal contract: legacy Neg Risk CTF Exchange v1**. Confidence: **high**. Polygonscan labels the address as [Polymarket: Neg Risk CTF Exchange](https://polygonscan.com/address/0xc5d563a36ae78145c45a50134d48a1215220f80a), shows it as a contract with verified source, contract name `NegRiskCtfExchange`, 0 POL balance, roughly 393k transactions, and creator `0xe9Ac97D2BE532C0de63Ec26270EA3F217E207326` in the [Nov-28-2023 deployment transaction](https://polygonscan.com/tx/0x5935e680b186f72b6db932ce351dd6f0288b1e056ab8d0d307c5146606879369). Its constructor arguments point to Polymarket's USDC.e collateral, CTF, Neg Risk Adapter, proxy factory, and safe factory on Polygonscan, which is contract-system wiring rather than a wallet/relayer profile ([constructor view](https://polygonscan.com/address/0xc5d563a36ae78145c45a50134d48a1215220f80a#code)). Polymarket's `contract-security` repo lists `NegRisk CtfExchange` at this exact address and `CtfExchange` at `0x4b...8982e` as Polygon mainnet deployments ([contract-security README](https://github.com/Polymarket/contract-security#polymarket-contract-security)). Polymarket's V2 migration docs show the Neg Risk verifying contract moving from `0xC5...f80a` to V2 `0xe222...0F59`, and current contract docs now list that V2 address as canonical ([migration docs](https://docs.polymarket.com/v2-migration), [contracts docs](https://docs.polymarket.com/resources/contracts)).

Evidence notes:

- Local fingerprint, `trades_seed.parquet`: `as_maker=0`, `as_taker=40,863,328`, first fill `2023-12-22 03:18:40`, last fill `2025-10-07 16:39:50`.
- Local fingerprint, delta shards: `as_maker=0`, `as_taker=61,354,579`, first fill `2025-10-07 16:39:52`, last fill `2026-04-28 11:00:40`.
- Literal web/Dune/X-style searches did not surface an independent relayer/operator identity; the indexed results consistently resolve this string to `NegRiskCtfExchange`, Polygonscan, or Polymarket contract-address references.

Strategic implication: **If we filter this address, we are excluding neg-risk-market v1 exchange-internal `OrderFilled` legs where the exchange contract is emitted as the taker, which means the filter removes a settlement/decoding artifact rather than any identifiable retail UI or batched smart flow.**

## Cross-address Comparison

These are the same kind of infrastructure, not two unrelated operators: `0x4b...8982e` is the standard-market CTF Exchange v1, and `0xC5...f80a` is the neg-risk CTF Exchange v1. Their shared fingerprint is decisive: both are verified Polymarket exchange contracts, both appear only as `taker` in the local fill data, and both stop in the local raw shards at `2026-04-28 11:00:40`, matching the CLOB v1 to v2 migration window. Current canonical V2 exchange addresses are `0xE111180000d2663C0091e4f400237545B87B996B` and `0xe2222d279d744050d28e00520010520000310F59`, so any forward-looking "exchange-internal leg" filter should be version-aware.

## Open Questions

- The local field named `transaction_hash` is one-to-one with these filtered rows in the available parquet fingerprints, so it was not useful for batching diagnostics; either the field is already event-unique in this dataset or the shards contain an expanded/normalized identifier.
- I did not get a true top-10 interaction counter from Polygonscan because the no-key API path is now blocked by Etherscan V2 API-key requirements; public profiles and source were enough for high-confidence classification.
- Discord/forum search was not accessible in the timebox. That does not materially weaken the classification because Polygonscan, Polymarket source, Polymarket contract-security, Polymarket docs, and the local fingerprint all agree.

## Recommended Next Action

Justin should relabel the denylist category from `relayer` to `exchange_internal_leg`, include both V1 and V2 exchange addresses, and rerun Block B / Block E Lite with a decoder-level maker/taker interpretation check before treating the lift as a trader-flow result.
