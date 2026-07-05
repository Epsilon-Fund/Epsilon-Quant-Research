# Observatory — future web-launch automation (SCAFFOLD ONLY, not wired)

> Status: **documented future switch, deliberately NOT live** (v3 mandate:
> attended/manual only, no paid key, no unattended cron, website parked).
> Nothing in this file runs today; it exists so flipping the switch later is a
> configuration exercise, not a design session.

## What the switch turns on

One unattended daily cycle: run the pipeline, regenerate the showcase artifacts,
upload them to the website. Everything already exists except the credentials and
the scheduler:

```
[n8n cron 13:00 UTC]
  └─ ssh/local exec:  cd polymarket/research &&
       GOOGLE_APPLICATION_CREDENTIALS=<repo>/secrets/newsfeed-epsilon-721cd7176ae8.json \
       GUARDIAN_API_KEY=... ANTHROPIC_API_KEY=... \
       PYTHONPATH=. uv run python -m newsagent.run_daily --stage all
  └─ on success: upload data/newsagent/showcase/{showcase.json,index.html}
       to the epsilon-webs1te ingestion point (colleague owns the site side;
       hand him showcase.json — the page is his to restyle)
  └─ on failure: notify (n8n error workflow → email/Telegram), do NOTHING else —
       the ledger must never receive a half-run's numbers twice (same-day republish
       is idempotent by design, so a manual rerun after a failure is always safe)
```

## Preconditions (in order)

1. **Extraction key** — Stage-A extraction (cheap model, batched, cached;
   steady-state ≈ tens of calls/day). Two provider paths behind the v3.1 flag
   (`--provider` / `NEWSAGENT_EXTRACT_PROVIDER`): paid `ANTHROPIC_API_KEY`
   (Haiku — the alpha-fit-era default) or free-tier `GEMINI_API_KEY`
   (2.5 Flash — run `scripts/newsagent_provider_spotcheck.py` against the Haiku
   cache before trusting it; alpha rides on the Haiku-era feature distribution).
   Both API code paths exist but have never been exercised against a live key —
   verify once attended before trusting them unattended.
2. `GUARDIAN_API_KEY` registered dev key (demo key in use).
3. `GOOGLE_APPLICATION_CREDENTIALS` exported in the runner env (key already in
   `secrets/`, gitignored).
4. Email credential for newsletter ingestion — **done since v3.1**: Gmail-API
   OAuth read-only (`secrets/gmail_client_secret.json` + `gmail_token.json`,
   consent minted 2026-07-05; refresh is automatic). The pipeline still degrades
   gracefully if the token is revoked.
5. IP-scrub re-check of the public artifacts once sources/markets change
   (`tests/test_newsagent_fv.py` carries the automated assertions).
6. Website ingestion agreed with the site colleague (he owns UX; we hand him
   `showcase.json`).

## Operational guardrails (carry into the n8n workflow verbatim)

- Ledger writes stay append-only through the sf CLI; the anti-post-hoc state
  machine is the single writer. Never bypass it from automation.
- Settlements (`sf settle`) and slate refreshes stay ATTENDED — they are
  judgment calls (resolution reading, market curation), not cron work.
- The α/γ refit (`scripts/newsagent_stageb_calibration.py --fit`) stays attended:
  every refit is documented in the findings note.
- Cost circuit breaker: cap the n8n run at one attempt/day; no retry storms into
  the paid API.
