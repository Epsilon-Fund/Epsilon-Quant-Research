---
title: "agent-reach as an evidence-source extension for the Calibration Observatory — install, doctor results, channel→market mapping, integration design, adopt/park verdicts, and the 2026-08-24 BUILD addendum + dateline AMENDMENT (shipped; the page channel now reaches two domains, and not the one we most wanted)"
created: 2026-08-24
status: SHIPPED 2026-08-24, then AMENDED the same day. Build slice live (worklist + --reach-file ingestion + two-level gate + chrome-strip + display boundary + declared weights). Justin ADOPTED the two-source dateline rule in his own variant (tie-break, not discard-on-disagreement) — locked at § 6a, implemented and measured at § 6b. It unlocks exactly ONE source family (state.gov/releases/) and NOT its priority-1 target: ukmto.org has a body dateline on every incident but no month anywhere in any URL, so it stays navigation-only. Reach items reaching packets went 0 -> 2. Two same-day documents were refused as future-dated, which makes the dateline path a T+1 channel — measured, not designed; the clamp repair is proposed, not taken. 636 tests green. No alpha refit, no forecast claim
owner: justin
project: polymarket
para: project
hubs:
  - strat_news_agent_showcase
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - news-agent
  - showcase
  - tooling
  - scoping
---
# agent-reach as an evidence-source extension for the Observatory — what it reaches that we don't, and what it would cost

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · Table terms: [[polymarket_table_dictionary]]
> Builds on [[newsagent_observatory_v32_findings]] (the shipped v3.2 Observatory) and the source/licence groundwork in [[newsagent_repo_data_radar_findings]]. This is a **scoping pass**: the tool is installed and probed, the integration is designed on paper, and **no repo code was changed**.

## Plain-English Summary

- **What this note is:** an evaluation of [agent-reach](https://github.com/Panniantong/agent-reach) (MIT) as a new *evidence source* for the Epsilon Calibration Observatory — the daily attended run that publishes our independent fair value vs the Polymarket mid for 24 politics/macro markets. The question asked was narrow: **does it reach evidence our current stack (Guardian + RSS + GDELT + newsletters + macro PDFs) cannot, and is that worth the fetch cost?**
- **What agent-reach actually is:** not a data API and not an MCP data server. It is an **installer + health-checker** that wires up free, keyless fetch paths (Jina Reader for any URL→markdown, Exa semantic search over a hosted MCP endpoint, yt-dlp for YouTube subtitles, feedparser for RSS, `gh` for GitHub) and then installs a Claude Code **skill** that tells an agent which shell command to run for which platform. The agent shells out directly; agent-reach itself fetches nothing at run time and caches nothing.
- **Install result on this machine:** clean. `uv tool install` from the GitHub archive (pipx is not installed here; see § Install record), then `install --system` for the keyless tier only. **5/15 channels report green**, of which exactly **three are relevant to us: web (Jina), video (YouTube/yt-dlp), search (Exa)**. Cookie/browser-session channels (Twitter, Reddit, Facebook, Instagram, XHS) were deliberately **not** configured — out of scope this pass, and Twitter needs a burner-account decision from Justin.
- **The honest headline:** the one genuinely new capability is **primary-source reach** — the actual page or transcript the market resolves on, rather than a journalist's account of it. Proven live during this pass: the July-2026 FOMC statement fetched from federalreserve.gov carries the decision sentence verbatim with a `Published Time` equal to the release instant (18:00:15Z), and the official Fed YouTube channel's 45-minute press conference yields a 6,753-word human-written caption track. Neither object exists anywhere in the current packet.
- **The honest cost:** the fetch is manual-ish, one-shot, uncached, and the upstreams break by design. Proven live during this pass: `r.jina.ai` **refused reuters.com outright** ("Anonymous access to domain www.reuters.com blocked… due to previous abuse") because the keyless tier is a shared anonymous pool; `nobelprize.org` came back with **no timestamp at all**; and 2 of 3 Exa results had **`Published: N/A`**. Every one of those is an item that must be *discarded*, not degraded into the packet.
- **Verdict:** **ADOPT** Jina (narrow, curated official-source URLs per market), **ADOPT-NARROW** YouTube transcripts (event-triggered only — FOMC pressers, State Dept briefings, hearings), **PARK** Exa as a discovery-only aid (never an evidence row), **PARK** the rest, **OUT OF SCOPE** all social channels.
- **SIGN-OFF (2026-08-24, Justin):** all three blocks answered — source weights **APPROVED as proposed** (official_primary 1.0 / official_state_claim 0.5 / official_transcript 1.0 / unknown_video 0.5 / Exa→RSP passthrough / state dashboards excluded); official primary documents **may display headline + source + link** with bodies internal (the `pdf_ingest` treatment); a **reach slot of ≤2 displacing Guardian 6→4** on reach days is approved; **burner Twitter = NO** (revisit only on demonstrated systematic under-coverage); skill housekeeping done (English reinstall, recorded in `skills-lock.json` as a **vendor-owned never-patch exception**, kept machine-wide). **Still nothing wired** — the weights bind the build whenever it happens, and approval is not a build.

## What agent-reach is, precisely (and what it is not)

Getting this wrong is the main way this evaluation could have been misread, so it is stated first:

| It is | It is not |
|---|---|
| A Python CLI (`agent-reach`) that **installs** third-party fetch tools and **reports** which of them work right now (`doctor --json`) | A data API, a feed, or a source of content in its own right |
| A **Claude Code skill** (`~/.claude/skills/agent-reach/`) containing a routing table: "for a web page run `curl https://r.jina.ai/URL`; for YouTube run `yt-dlp --write-sub …`" | An MCP data server. It ships an *optional* MCP wrapper, but the documented path — and the one we would use — is the agent shelling out directly |
| A thin, keyless convenience layer over tools that are individually free and individually installable | A scheduler, a cache, a monitor, or a retry/backoff layer. **There is no persistence of any kind** — every fetch is one-shot |

**Consequence for us:** adopting agent-reach is really adopting *three shell commands plus a health check*. That is a feature, not a criticism — it means the integration surface is tiny and the dependency is trivially reversible (`agent-reach uninstall`). It also means **every durable property we need — caching, timestamps, dedupe, display flags, source weights — has to be ours**, in the existing packet store. Nothing about the tool supplies them.

## Install record and setup burden

Everything below was run read-only first, then with `--system` for the keyless tier only. No cookies, no logins, no account touched.

**Deviation from the brief (flagged, not silent):** the brief said `pipx install`. **pipx is not installed on this machine** and installing it would itself be a system change. `uv tool install` is the exact functional equivalent (isolated venv + shim on PATH) and is what [[COWORK]] § Repo conventions mandates anyway ("`pip install` is never to be run directly; use uv"). The source is still the GitHub archive URL, **not** the same-named PyPI package (which is a different project):

```bash
uv tool install "https://github.com/Panniantong/agent-reach/archive/refs/heads/main.tar.gz"
agent-reach install --env=auto            # read-only check (safe mode is the default)
agent-reach install --env=auto --system   # keyless tier only — no --channels
agent-reach doctor --json
agent-reach skill --install
```

Installed **v1.5.0** (repo `Panniantong/Agent-Reach`, MIT, 74.6k stars, last commit 2026-08-12). 21 packages into an isolated uv tool venv; ~2 seconds; no build failures on Python 3.14.

**What `--system` wrote to this machine** (audited, not assumed — the code path was read before it was run):

| Path | What | Reversible |
|---|---|---|
| `~/.local/share/uv/tools/agent-reach/` | The isolated venv + `agent-reach` shim | `uv tool uninstall agent-reach` |
| `~/.config/yt-dlp/config` | Appended one line: `--js-runtimes node` (YouTube needs a JS runtime) | delete the line |
| npm global (`~/.local`) | `mcporter` (the MCP client Exa is called through) and `undici` (Node proxy support) | `npm rm -g mcporter undici` |
| `~/.mcporter/mcporter.json` | Registers one server: `exa → https://mcp.exa.ai/mcp`. No key, no token | delete the entry |
| `~/.agent-reach/tools/` | Created, **empty** — a staging dir for optional-channel helpers we did not install | `agent-reach uninstall` |
| `~/.claude/skills/agent-reach/` | `SKILL.md` (6.8 KB) + `references/*.md` (7 files, 22 KB) | delete the dir |
| `~/.agents/skills/agent-reach/` | The same skill again — the installer writes to **every** skill root that already exists | delete the dir |

It did **not** install `gh` or Node — both were already present. Had either been missing, the macOS path would have shelled out to Homebrew, which is **not** on this machine, and `--system` would have failed with an explicit message rather than silently doing something else.

**Setup burden: low, but three gotchas worth recording.**

1. **The skill installs at user level, machine-wide** — into `~/.claude/skills/` and `~/.agents/skills/`, so it is live in *every* Claude Code session on this machine, including crypto/live-trading sessions that have nothing to do with news. See § Skill-conflict concerns.
2. **With `LANG`/`LC_ALL` unset (the state of this machine) the installer picks the Chinese `SKILL.md`, not `SKILL_en.md`.** The English variant exists and is a straight swap: `AGENT_REACH_LANG=en agent-reach skill --install`. Left as-installed pending Justin's call.
3. **The skill installer is `force=True`: it `rmtree`s the target directory and rewrites it on every `install` / `skill --install`.** Any local patch we ever applied would be destroyed without warning — unlike the repo's vendored skills, which are pinned and hashed in `skills-lock.json`.

## Doctor results on this machine (2026-08-24)

`agent-reach doctor --json`, verbatim status per channel. **Tier 0** = keyless/zero-config, **tier 1** = needs a login/cookie/extra CLI, **tier 2** = manual only.

| Channel | Status | Tier | Active backend | Relevant to the Observatory? |
|---|---|---|---|---|
| `web` (any URL → markdown, Jina Reader) | **ok** | 0 | Jina Reader | **Yes — the main find** |
| `youtube` (video metadata + subtitles) | **ok** | 0 | yt-dlp | **Yes — pressers/hearings/debates** |
| `rss` (RSS/Atom) | **ok** | 0 | feedparser | No — duplicates `feeds.RSS_FEEDS` |
| `v2ex` | ok | 0 | public API | No |
| `bilibili` | ok | 1 | search API (search only) | No |
| `exa_search` (semantic web search) | warn | 0 | — (configured, unverified) | **Partially — discovery only** |
| `github` | warn | 0 | — (installed, unverified) | No |
| `twitter` | warn | 1 | — | Out of scope (burner decision) |
| `linkedin` / `xueqiu` | warn | 1–2 | — | No |
| `reddit` / `facebook` / `instagram` / `xiaohongshu` / `xiaoyuzhou` | off | 1 | — | Out of scope (cookies) |

**Column meanings.** *Status*: `ok` = the backend binary/endpoint is present and the channel's own smoke check passed; `warn` = installed/configured but deliberately **not** verified end-to-end; `off` = no backend at all. *Active backend*: which of several possible tools would actually serve this channel right now.

**Read — and this is the load-bearing point of the whole doctor section: `doctor` is a necessary gate, not a sufficient one.** All three of the following were established by hand during this pass:

- `github` reports **warn** while `gh api repos/...` works perfectly (doctor refuses to run `gh auth status` because it writes a device-id; the warn is a deliberate false negative).
- `exa_search` reports **warn** while `mcporter call exa.web_search_exa` returned three results in one shot (doctor won't call the remote endpoint just to check).
- `web` reports **ok** while `r.jina.ai` **refused reuters.com with HTTP 403** (see below). Green at the channel level says nothing about the specific URL you are about to fetch.

The design in § Integration therefore uses doctor as a **cheap pre-flight**, and adds a **per-item validation gate** that decides whether a fetched object is real evidence.

## Live probes — what actually came back

Doctor status is a claim; these are measurements. All read-only, all public, no accounts.

### Jina Reader (`web`) — the primary-source path

**Probe 1 — the market's literal resolution document.** `curl -s https://r.jina.ai/https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm`

```
Title: Federal Reserve issues FOMC statement
URL Source: https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm
Published Time: Wed, 29 Jul 2026 18:00:15 GMT
...
The Committee decided to maintain the target range for the federal funds rate at 3-1/2 to 3-3/4 percent…
```

That sentence *is* the resolution of `will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting`. Nothing in the current stack ever sees it — the packet sees Guardian and RSS *coverage* of it.

**Probe 2 — where does `Published Time` come from?** Compared against the origin's HTTP headers on two pages:

| Page | Jina `Published Time` | Origin `Last-Modified` | Match |
|---|---|---|---|
| `…/pressreleases/monetary20260729a.htm` (immutable dated doc) | Wed, 29 Jul 2026 18:00:15 GMT | Wed, 29 Jul 2026 18:00:15 GMT | exact |
| `…/monetarypolicy/fomccalendars.htm` (rolling index page) | Wed, 19 Aug 2026 18:00:38 GMT | Wed, 19 Aug 2026 18:00:38 GMT | exact |

**Read:** Jina's `Published Time` is the origin's `Last-Modified`, nothing more. For an **immutable dated document** that equals publication and is a defensible evidence timestamp. For a **rolling page** it is "when the page was last touched" — the calendar page's 19-Aug stamp describes an edit, not an event, and using it as an item date would be a lookahead violation waiting to happen. This distinction drives the discard rule in § Timestamp discipline.

**Probe 3 — the failure modes, all three hit on the first attempt:**

- `reuters.com` → **HTTP 403** with a JSON body: `AbuseAlleviationError … Anonymous access to domain www.reuters.com blocked until Mon Aug 24 2026 12:36:25 GMT … DDoS attack suspected: Too many requests`. The keyless tier is a **shared anonymous pool**; a domain can be blocked by other people's traffic, with a rolling unblock time we neither control nor can predict.
- `nobelprize.org/prizes/peace/` → HTTP 200, content fine, **no `Published Time:` line at all**. Untimestamped.
- `state.gov/briefings/` → **HTTP 200 carrying a 404 page**, with `Warning: Target URL returned error 404: Not Found` in the preamble. A naive fetch would have put a "Page not found" page into the packet as evidence.

**Read:** the failures are not exceptional, they are the normal operating regime. Two of the three return **HTTP 200 or a well-formed body** — so "the curl succeeded" is worthless as a success test. This is what makes the per-item validation gate mandatory rather than nice-to-have.

**Probe 4 — cost shape.** The FOMC statement fetch returns **52 KB of markdown, of which the statement itself is ~2.25 KB (4.3%)**; the rest is site navigation chrome. Latency ranged 0.4 s (warm) to 7.9 s. **Read:** un-stripped, a single official page would cost ~13k tokens of navigation junk to extract features from. A chrome-strip step is not optional — the `pdf_ingest` disclaimer-strip is the existing precedent for exactly this.

### YouTube (`video`) — pressers, hearings, debates

`yt-dlp --write-sub --write-auto-sub --sub-lang en --skip-download` on the official Federal Reserve channel's *FOMC Press Conference, July 29, 2026* (`DLFXUkOc_7I`):

| Field | Value | Why it matters |
|---|---|---|
| Caption kind | `Kind: captions` (human-written, not ASR) | Official-channel pressers ship real transcripts, not error-prone auto-captions |
| Size | 68 KB VTT → **6,753 words** of text (~9k tokens) | One transcript is roughly the size of an entire day's 12-article packet |
| `timestamp` (upload) | 1785356671 = **2026-07-29T20:24:31Z** | ~2h24m *after* the 18:00:15Z statement, ~1h after the presser ended |
| `duration` | 2704 s (45 min) | Needed for the live-stream timestamp rule below |

A third-party re-stream of the same event showed the `was_live` split: `release_timestamp` 17:57:02Z (stream start, *before* the event) vs `timestamp` 19:16:16Z (upload). **Read:** for a live item, the start-time field predates the content; dating a transcript by it would import information from before it existed. The conservative rule is in § Timestamp discipline.

### Exa (`search`) — semantic retrieval

`mcporter call exa.web_search_exa query="Strait of Hormuz shipping traffic tanker transits normal" numResults=3`:

| Result | Published field | What it is |
|---|---|---|
| `tankermap.com/analytics/straits/hormuz` | **N/A** | A live rolling dashboard (vessels currently in zone) — state, not event |
| `reuters.com/…/shipping-via-hormuz-strait-slows-after-tanker-attacks-data-shows-2026-08-16/` | `2026-08-16T00:00:00.000Z` | A real wire story — **and Reuters has no public RSS, so the current stack cannot reach it** |
| `lloydslistintelligence.com/…/strait-of-hormuz-brief-19-august-2026` | **N/A** | Trade-press brief (paywalled) |

**Read:** Exa genuinely surfaces things Guardian+RSS miss (Reuters, trade press) — but **two of three rows carry no date**, and the one that does is **date-only** (midnight granularity). Exa returns *highlights* (extracted snippets of the source), which is licensed third-party text, not a headline. It therefore fails both the timestamp test and the display test as an evidence source. Its honest role is **URL discovery**: find the page, then fetch it through the Jina path and apply the Jina rules — noting that the Reuters URL it found is precisely the one Jina is currently blocked from.

## Channel → market mapping

The slate is the 24 markets in `newsagent/config.py`. Grouped by family, with the concrete gap each healthy channel fills. "Current reach" means Guardian search + the 7 RSS feeds + GDELT attention + ING/Bloomberg newsletters + the five macro PDFs.

| Family (markets) | Current reach | What Jina adds — concrete URL | What YouTube adds — concrete channel | Exa's role |
|---|---|---|---|---|
| **US macro / Fed** (`…fed-interest-rates…july-2026`, `…september-2026`) | Guardian/RSS coverage of the decision; ING + Bloomberg + the 5 bank PDFs give the rates view | `federalreserve.gov/newsevents/pressreleases/monetary<YYYYMMDD>a.htm` — the statement itself, plus `…a1.htm` (implementation note) and `fomccalendars.htm` for the meeting date | Official Fed channel: the full presser (proven: 6,753-word human caption track) — the *guidance language*, not just the decision | Low value; the sources are known and fixed |
| **Hormuz** (`…july-31`, `…december-31`) | Thinnest family in the slate — Guardian covers escalation, not "traffic returned to normal" | UKMTO advisories (`ukmto.org`), IMO circulars, CENTCOM/5th-Fleet releases — the dated, official navigational record the resolution actually tracks | Rare; occasional CENTCOM briefings | **Highest value here** — this is the family where we do not know the right URL yet |
| **Iran / US–Iran** (`…regime fall`, `…us invade iran`, `…diplomatic meeting by july 17`, `…MOU withdrawal`) | Guardian + RSS + WP-CE | `state.gov` readouts + press-briefing transcripts, `iaea.org` press releases, Iranian MFA statements — a "did they meet" market resolves on a **readout**, not on coverage | Official State Dept channel: daily press briefings (a question about a meeting is usually answered live at the podium before any wire story) | Discovery for the Iranian-side sources |
| **Russia / Ukraine** (`putin-out`, `zelenskyy-out`, `…recapture crimea`, `…ceasefire`, `united-russia…`) | Guardian + RSS + GDELT bursts | `kremlin.ru` and `president.gov.ua` readouts, NATO releases — **but see the state-actor caveat below**: these are primary *and* interested | Occasional official addresses | Discovery |
| **US domestic politics** (`trump-out`, `dems-house`, `dems-senate`, `becerra…`, `…wealth-tax`) | Guardian + Politico + The Hill (strong here); two of these are already tagged **data-driven** (polling-dominated) | `congress.gov` resolution/vote records for the impeachment leg of `trump-out`; state SoS filing pages for ballot questions | Official House/Senate committee channels: hearings | Low — Politico/The Hill already cover this densely |
| **Other geopolitics** (`china-taiwan`, `netanyahu-next-pm`, `us-invade-cuba`) | Guardian + RSS | Knesset/Israeli government releases; MoD/State releases | Rare | Discovery |
| **Awards / intl elections** (`trump-nobel`, `lula-brazil`, `bardella-france`) | Guardian + RSS | `nobelprize.org` announcement page (**proven untimestamped — a worked example of the discard rule**); TSE Brazil and French Interior Ministry results pages | The Nobel announcement is livestreamed on the official channel — a dated artifact where the page is not | Discovery |

**Practical example, end to end.** On 2026-07-29 the Observatory's Fed-July card would have been formed from Guardian/RSS stories written *about* the decision, each entering Stage A as a headline + trail + lede. Under the proposed extension the attended session would additionally fetch two objects: the statement page (2.25 KB after strip, timestamped 18:00:15Z, containing "decided to maintain the target range… at 3-1/2 to 3-3/4 percent") and the presser transcript (9k tokens, timestamped 20:24:31Z). The first is *decisive* evidence in the Stage-A sense — it is the resolution fact, not a report of it. The second is *guidance language* — what Stage A would score as stance/strength for the September market. Neither is displayed on the public page under the rules below; both are counted.

**The honest limit, stated plainly:** the two Fed markets are tagged **data-driven** in `config.py` — rate odds price off Fed-funds futures, and our cards already say we are structurally blind there. Primary-source reach does **not** repeal that tag. It improves the *evidence* on a market whose dominant channel is still not news text. Claiming otherwise would be exactly the kind of over-reach [[CODEX]] § Realism calibration exists to stop.

## Integration design (design only — nothing wired)

The Observatory runs **attended**, out-of-band, by a human-supervised Claude Code session ([[newsagent_observatory_v31_findings]], [[newsagent_observatory_v32_findings]]). The proposal keeps it that way: **no new backend fetcher module, no new dependency in `feeds.py`, no cron.**

### Where it slots in

`run_daily.py` already has a well-established **out-of-band ingestion idiom**: `--stage onboard --priors-file`, `--stage extract --features-file`. Agent-produced JSON is written to disk and ingested by an explicit flag. The reach path should be the **third instance of that same idiom**, not a new mechanism:

1. `--stage fetch` runs as today (Guardian + RSS + newsletters + PDFs + GDELT) and additionally writes a **reach worklist** per market: the curated official-source URLs for that market, plus any event trigger (e.g. "FOMC day → presser video").
2. The **attended agent** (using the agent-reach skill) executes the shell fetches, applies the validation gate, and writes article-shaped dicts to `data/newsagent/live/<date>/reach_items.json`.
3. `--stage fetch --reach-file reach_items.json` (or a re-run that picks the file up if present) merges them into the packets through the **existing** `_keyword_filter` → `build_packet` path. Stage A, Stage B, the ledger and the dashboard need **zero** changes beyond the display filter already being enforced.

Item shape is the existing article dict — `title`, `seendate`, `domain`, `url`, `trail`, `lede`, `scan_text`, `display` — so the stripped body goes in `scan_text` (the `pdf_ingest` precedent: a Fed mention on page 3 still matches the rates keywords) and nothing downstream learns a new schema.

**Two mechanical catches found while reading the code — both are load-bearing:**

- **Stage-A cache collision.** `features.cache_key(slug, title)` is `sha1(slug + "|" + title)` plus the prompt version. A rolling page fetched on two different days produces the **same title with different content** and would silently return day-1 features for day-2 content. Rule: a reach item's title must be content-unique (append the item's `seendate` or a content hash), **or** the item must be an immutable dated document (whose title is unique per event by construction). This alone is a reason to prefer dated documents over rolling pages.
- **Packet slot budget.** `build_packet` caps at 12 items with per-slot caps (newsletters ≤2, PDFs ≤2, Guardian ≤6, RSS ≤4, WP fills). A reach slot must be given an explicit cap and it **displaces** something. Proposal: **reach ≤2** (official document + transcript), taken out of the Guardian slot (6→4) on days when reach items exist, because a primary document dominates three restatements of it. This is a declared choice, not a fitted one, and it changes what Stage B sees — so it belongs in the sign-off block.
- **Window.** `_keyword_filter` windows items to the packet window (72h/7d), with PDFs already given a special 8-day window. Official documents are *state-of-record*, not news: propose **14 days** for official documents and **72h** for transcripts. Declared, not fitted.

### (a) Timestamp discipline — lookahead

Same spirit as the GDELT rule (`gdelt_bq.py`: the server's end-bound silently leaks ~24h, so a **client-side `seendate` filter is mandatory** and any item with `seendate > cutoff` is dropped). Every new channel gets the same treatment: **a defensible timestamp or it is not evidence.**

| Channel | The timestamp we use | Where it comes from | Discard the item when |
|---|---|---|---|
| **Jina page fetch** | The `Published Time:` header — empirically **identical to the origin's HTTP `Last-Modified`** (verified on two federalreserve.gov pages) | First ~6 lines of the r.jina.ai response | (1) the header is **absent** (nobelprize.org case); (2) the URL is a **rolling/index page**, where `Last-Modified` is a touch-time, not an event time (fomccalendars.htm case) — such a page may only be read as undated *state*, never as a dated evidence item; (3) the timestamp is **after the run cutoff** |
| **YouTube transcript** | `max(timestamp, release_timestamp + duration)` — i.e. **the later of upload time and stream-end** | `yt-dlp --dump-json` | (1) the computed instant is after the cutoff; (2) `live_status` is `is_live` or `post_live` (content still accruing); (3) **never** date a transcript by the event date in its title |
| **Exa result** | None that we trust — `Published` was N/A on 2 of 3 probe rows, and date-only when present | Result header | **Always**, as an evidence row. Exa may only produce a *URL*, which must then independently pass the Jina rule. If a date is present and date-only, treat it as 23:59:59Z of that day (conservative) |
| **RSS via feedparser** | Unchanged from `feeds._rss_dt` | Feed entry | Unchanged |

**Why "the later of" for video.** A transcript covers the whole event, so the information in it is only complete at the end of the stream; dating it at stream *start* would let post-cutoff content into a pre-cutoff packet. Worked example: the third-party FOMC re-stream carried `release_timestamp` 17:57:02Z — three minutes *before* the 18:00:15Z statement. A packet cut at 18:00Z that accepted the stream-start date would have ingested a transcript containing the decision it was supposed to be blind to. That is the exact failure the GDELT seendate rule exists to prevent.

**Live-only by default.** RSS, newsletters and PDFs are already excluded from historical reconstruction by construction (no timestamped archive). Reach items inherit that default — **with one deliberate exception**: YouTube uploads and immutable dated documents *do* have timestamped archives, so they may enter `backfill-fetch` reconstruction, but only under the rules above and only when the fetch is re-run with an explicit as-of cutoff. Any reach item without a defensible timestamp is barred from reconstruction absolutely, and if a refit ever consumes reconstructed packets, this is the first invariant to re-test.

### (b) Display rules — privacy and licence

The `email_ingest.py` pattern is the model: private/licensed items carry `display=False`, the public page shows only a generic label and a **count** ("private analysis item — not displayed"), and the filter is enforced **at the page boundary**, with tests. That boundary placement is not incidental — v3.1 caught a latent leak precisely because the filter existed at the source but not at the page ([[newsagent_observatory_v31_findings]]).

| Item type | Display treatment |
|---|---|
| **Jina-fetched page text (default)** | `display=False`. Counted-never-shown, generic label only. Body is Stage-A input exclusively |
| **Jina fetch of a wired outlet** (theguardian.com, bbc.co.uk, news.sky.com, politico.com, thehill.com) | Headline + link only, as today. The **fetched body is never displayed** — display rights cover the headline/link, not our full-text copy |
| **Official primary documents** (federalreserve.gov, state.gov, congress.gov, nobelprize.org, ukmto.org, iaea.org) | **APPROVED 2026-08-24**: headline + source + link, aggregator-style, **body internal-only** — the exact treatment `pdf_ingest` already gives bank research PDFs. (Was `display=False` pending sign-off.) |
| **Any transcript, any source** | `display=False`, always. A caption track is a full copy of the work; even where the underlying work is a public-domain US-government product, re-hosting it adds nothing to a page whose job is showing our *number*. The card may say "1 transcript read (not displayed)" |
| **Exa highlights** | Never displayed and never stored as an evidence row — they are extracted snippets of third-party pages |

**Test obligations if this is ever built** (mirroring the existing 82-test suite): a reach item with `display=False` must never appear in the feed pane or a card's evidence list; `n_private` must increment; the IP-scrub end-to-end test must pass with reach items present; a transcript item must be unreachable from the page regardless of its domain.

### (c) Source weighting — Scheme-A status + lean, proposed neutral

New source *types* need a declared status→weight row before they can influence Stage B, exactly as ING/WP-CE did in v3. `sourceweights.get_weight()` currently returns the neutral `UNCOVERED_W = 1.0` for anything outside the RSP tier map, and `sourcelean.get_lean()` returns `None` (multiplier 1.0) for unrated domains — so **doing nothing is already the neutral default**, and that is what these run at until Justin signs off.

The proposed rows are in the sign-off table at the bottom. One substantive point deserves stating here rather than in a table cell: **"official" is not the same as "reliable."** A Federal Reserve statement about the Federal Reserve's own decision is a procedural fact about the issuing institution — as close to ground truth as a source gets. A Kremlin readout about Russian intentions is a primary document *and* an interested party's contested claim about itself. Collapsing both into one "official = 1.0" row would import propaganda at maximum weight. The proposal therefore splits official sources into **procedural** (1.0) and **state-claim** (0.5), which is a declared judgment, not a fit — n is far too small to fit, same honesty as the RSP and lean multipliers.

### (d) Fragility — the doctor gate and graceful absence

agent-reach's upstreams break routinely **by design** (free tiers, shared anonymous pools, scraper-vs-platform arms races). The gate mirrors `_refresh_gdelt` exactly — which prints `gdelt: skipped (<why>)` and lets the burst feature degrade to `None` with the gamma term inert, never breaking the daily run.

**Two-level gate:**

1. **Pre-flight (channel level).** Once per run, at the top of `--stage fetch`: run `agent-reach doctor --json` (5 s timeout), cache it into the day dir as `reach_doctor.json`, map channel→status. Any channel not `ok` → skip its fetches, print `reach: <channel> skipped (<doctor message>)`, build the packet without it. A missing `agent-reach` binary is just another skip — the pipeline must run identically on a machine that never installed it.
2. **Per-item validation (the part doctor cannot do).** Because green ≠ working — proven three times above — every fetched object is validated before it becomes an item:
   - **Jina**: the response must not be a JSON body carrying a `"code": 4xx` (the Reuters 403 case); must contain a `Title:` line; must not contain `Warning: Target URL returned error` (the state.gov 404-as-200 case); and must yield ≥ N characters after chrome-strip.
   - **yt-dlp**: a `.vtt` must exist and yield ≥ ~200 words after cue-stripping (an empty caption response is a known transient, not proof the video has no captions).
   - **Exa**: at least one row with a URL; the row itself is never stored.
   - Any failure → the item is **absent**. Never a placeholder, never an error string stored as page text. The Reuters probe is the cautionary case: a naive fetch would have written `AbuseAlleviationError: … DDoS attack suspected` into the packet as evidence text and Stage A would have dutifully extracted features from it.

**Caching is ours.** agent-reach has none. Reach items live in the existing day dir (`data/newsagent/live/<date>/`) alongside `rss_cache/`, `wp_cache/`, `pdf_cache/` and the Stage-A `feature_cache/`, so a re-run of a day costs nothing and the append-only discipline is unchanged.

## Adopt / park verdict per channel

Cost is stated in **per week of attended runs (~7 runs)**, and assumes the fetch is manual-ish and one-shot.

| Channel | Verdict | What it adds per week | Honest cost |
|---|---|---|---|
| **web (Jina)** | **ADOPT — narrow** | The resolution document itself for the ~8–12 markets that have one (Fed statements, State readouts, UKMTO advisories, congress.gov records). This is the only genuinely new *kind* of evidence in the whole evaluation | ~60–85 fetches/week at 0.4–8 s each ≈ 1–2 min per run. **Chrome-strip is mandatory** (4.3% signal on the Fed page) or ~13k junk tokens per page; stripped ≈ 600 tokens/page, ~7k tokens/run. Per-domain anonymous blocks will happen and must degrade silently. The real cost is **attended judgment**: choosing the right URL per market once, then maintaining that list |
| **video (YouTube)** | **ADOPT — narrow, event-triggered only** | 1–3 transcripts: FOMC pressers (8/yr), State Dept briefings, hearings/debates. Never a daily sweep | ~9k tokens per transcript (6,753 words measured) — one transcript ≈ a whole day's packet. ~30 s fetch. Needs an explicit event trigger in the worklist, otherwise it is pure token burn on days when nothing happened |
| **search (Exa)** | **PARK — discovery aid only, never a source** | Finds URLs the current stack cannot reach (Reuters, Lloyd's List trade press), which matters most for the Hormuz family where we do not yet know the right official URL | 2 of 3 results are undated and its highlights are licensed snippets → fails both the timestamp and display tests as evidence. Use only when a packet is thin (`n_relevant < 3`), a handful of queries per week, and only to produce a URL for the Jina path — which may itself be blocked (the Reuters case is exactly this) |
| **rss (feedparser)** | **PARK** | Nothing. `feeds.RSS_FEEDS` already covers BBC/Sky/Politico/The Hill with a live-probed stdlib parser | feedparser is more tolerant than `xml.etree`, but that is a ~20-line swap we would not adopt a tool for |
| **github (gh)** | **PARK** | Nothing — no market in the slate resolves on repo activity | — |
| **v2ex / bilibili / xueqiu / xiaoyuzhou** | **PARK** | Nothing for this slate (Chinese-platform channels). `xueqiu` would only matter if an equities-linked market returned, and that branch is closed ([[od_equities_index_pricing_scope_findings]]) | — |
| **twitter / reddit / facebook / instagram / linkedin** | **OUT OF SCOPE this pass** | Potentially the fastest primary channel in geopolitics (officials post before wires) | Cookie/browser-session based, ToS-risky, account-linked. **Twitter requires a burner-account decision from Justin — flagged, deliberately not configured.** Nothing was installed, no browser profile was read |

**Aggregate read.** Adopting Jina + event-triggered YouTube costs roughly **1–2 minutes and ~10–20k extra tokens per attended run**, plus a one-time curation pass to pin the official URL(s) per market. That is cheap. What it buys is not "more articles" — the packet is capped at 12 and already full — it is a **different class of evidence**: the document the market resolves on, at the instant it was published. The counterweight is honest: it does not move the two data-driven Fed markets into news-tractability, it will fail unpredictably on high-value news domains, and every item it produces is `display=False` by default, so **none of it is visible on the public page** — it changes the number, not the show.

## Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** That Jina's `Published Time` == origin `Last-Modified` generalises beyond the two federalreserve.gov pages verified here (it is Jina's documented behaviour, but it is two data points); that `max(upload, release+duration)` is a conservative transcript timestamp in all live/VOD combinations; that official-channel captions stay human-written rather than silently degrading to ASR; that the proposed weight rows (procedural 1.0 / state-claim 0.5 / unknown 0.5) are declared judgments, not fits — n is far too small to fit, same posture as the RSP and lean multipliers; that a 2-item reach slot displacing 2 Guardian items is a net evidence gain (plausible, unmeasured).

**Live-only unknowns.** How often the keyless Jina tier blocks a domain we actually need, and for how long (one block observed, no base rate); whether Exa's date coverage improves for news-domain queries; whether primary-source evidence measurably improves forward Brier — **this is the only question that matters and it cannot be answered offline**: it needs settled forecasts under the extension, refit through the existing `scripts/newsagent_hist_backfill.py --fit` path; whether the ~9k-token transcripts change Stage-A feature quality or just add noise; whether the reach worklist stays maintainable as the slate refreshes.

**Power honesty.** Nothing in this note is a measurement of forecast quality. Every claim here is about *reach* (can we fetch object X, does it carry a defensible date) — not about *edge*. Under the [[newsagent_v0_gate_findings]] closure the fair-value-beats-the-mid claim stays closed and displayed; a better evidence path does not reopen it, and no result here should be cited as if it did.

## Sign-off table for Justin — ANSWERED 2026-08-24

**All three blocks are decided.** The v3 dependency these rows waited on also cleared the same day: the Scheme-A uncovered-source weights are **APPROVED and LIVE** (ING 0.9 / Wikipedia Current Events 0.8 / Bloomberg 0.9 / bank research desks 0.9 / genuinely unknown **0.5**), so the "unknown" rows below inherit **0.5**, not 1.0. The reach rows themselves are approved *as proposed* but stay **inert until the build**: no reach item can carry a weight before there is a reach item, and α is refit through `scripts/newsagent_hist_backfill.py --fit` whenever the weighting of real evidence changes (the activation on 2026-08-24 moved α 2.7 → 2.85; see [[newsagent_observatory_v33_findings]]).

### 1. Source weights — new source types

| Source type | Example domains | Proposed status | Proposed `w_rel` | Lean | Rationale | **Decision — Justin, 2026-08-24** |
|---|---|---|---|---|---|---|
| Official primary — **procedural** fact about the issuing institution | federalreserve.gov, congress.gov, nobelprize.org, ukmto.org, iaea.org, state.gov (readouts) | `official_primary` | **1.0** | unrated (×1.0) | The institution *is* the resolution source; a procedural fact about its own act is as close to ground truth as evidence gets | **APPROVED as proposed** (1.0) — activates with the build. |
| Official primary — **contested self-claim** by a state actor | kremlin.ru, mfa.gov.ir, president.gov.ua, mod.ru | `official_state_claim` | **0.5** | unrated (×1.0) | Primary ≠ disinterested. "Official" must not import a party's claims about its own contested conduct at full weight | **APPROVED as proposed** (0.5) — the procedural/state-claim split stands: 'official' must not import a party's contested claims about its own conduct at full weight. |
| Official-body **video transcript** (official channel) | youtube.com/@federalreserve, @statedept, House/Senate committee channels | `official_transcript` | **1.0** | unrated (×1.0) | Same standing as the institution's own text. Always `display=False` | **APPROVED as proposed** (1.0), always `display=False`. |
| **Third-party video transcript** | any other uploader | `unknown_video` | **0.5** (or the uploader's RSP tier where it maps to a wired outlet) | per outlet | Matches the pending "unknown 0.5" proposal | **APPROVED as proposed** (0.5, or the uploader's RSP tier where it maps to a wired outlet). |
| Exa-discovered page on an **RSP-covered** outlet | reuters.com, apnews.com, ft.com | existing RSP tier | RSP tier | RSP + AllSides | No new rule — the existing path reached through a new door | **APPROVED — passthrough to the existing RSP tier**, no new rule; Exa remains discovery-only and never becomes an evidence row itself. |
| Exa-discovered page on an **uncovered** domain | lloydslistintelligence.com | `unknown` | **0.5** | unrated | Matches the pending "unknown 0.5" proposal | **APPROVED** — inherits the now-approved `unknown` = **0.5**. |
| **State/data dashboards** (no event, no date) | tankermap.com | — | **EXCLUDE from Stage A** | — | State-not-event, no defensible timestamp. May inform tractability tags only, never an evidence row | **APPROVED — EXCLUDED from Stage A.** State-not-event, no defensible timestamp; may inform tractability tags only. |

**Also needed a yes/no — both ANSWERED 2026-08-24:**

- **May official primary documents display headline + source + link?** **YES** — they get the `pdf_ingest` treatment: headline + source + link on the public page, **bodies stay internal** (our fetched full text is never displayed, exactly as with Guardian text and bank-research PDFs). Transcripts are unaffected: they stay `display=False`, always.
- **Is a reach slot of ≤2 items, displacing Guardian 6→4 on reach days, acceptable?** **YES, approved as proposed.** A primary document dominates three restatements of it. This is a declared choice, not a fitted one, and it changes what Stage B sees — so when the build lands it must be reported as a packet-composition change, with α refit through the existing path.

### 2. The burner-Twitter question

The Twitter/X channel is cookie-based: it requires extracting a logged-in session from a real browser profile, which means an account is attached to our automated fetches. **Not configured, deliberately.** The decision Justin owns: **do we create a burner X account for this?**

- **For:** officials and agencies post to X before wires pick it up; for shock-type geopolitics markets (Hormuz, Iran, Russia/Ukraine) that lead time is the whole ballgame.
- **Against:** it is automated access against a ToS that prohibits it, on an account that can be banned; it puts a credential in `secrets/`; and the packet is capped at 12 items that are already competing. The current stack is not obviously *late* on any market in the slate — it is *shallow*, which primary-source Jina fixes more cheaply and with no account risk.
- **Recommendation: no, not now.** Revisit only if a specific market family demonstrates systematic under-coverage that a wire/official source could not have supplied in the same news cycle.

> **DECISION — Justin, 2026-08-24: NO.** No burner X account. The channel stays unconfigured and no browser profile is read. The single reopening condition is the one above and nothing else: a market family that demonstrates **systematic** under-coverage which a wire or official source could not have supplied in the same news cycle. "It would have been faster on X" for one event is not that; the evidence has to be a pattern across resolutions, and it should be argued from the settled ledger, not from a memorable near-miss.

### 3. Skill-conflict concerns (repo `.claude/skills/`) — all six answered 2026-08-24

| # | Concern | Detail | Decision needed | **Decision — Justin, 2026-08-24** |
|---|---|---|---|---|
| 1 | **Machine-wide trigger** | The skill installs to `~/.claude/skills/` **and** `~/.agents/skills/` — user-level, so it is live in every Claude Code session on this machine, including crypto/live-trading sessions | Keep, or uninstall the skill and drive the three shell commands directly (the CLI works without it) | **KEEP machine-wide.** The skill stays installed at user level in both skill roots; the CLI would work without it, but the routing table is the useful part and the cross-session presence is accepted. |
| 2 | **Trigger-greedy description** | Its frontmatter says *MUST USE* for research/search/"any URL", and the body instructs the agent not to invent its own approach when the skill is present. That competes with this repo's own routing law | Accept, or trim the description locally — but see #4 | **ACCEPT as-is** — and note it is *unpatchable* anyway (see #4): trimming the description locally would be destroyed on the next install. This repo's own routing law (CODEX/Sherpa) still governs what we actually run. |
| 3 | **Sherpa is safe; native triggering is not** | `tools/sherpa.py` indexes repo-local skill dirs only unless `include_global=True`, so agent-reach will **not** appear in Sherpa output. The overlap is purely with Claude Code's native description-triggering | Informational — no action unless #1 changes | **Informational, no action** — Sherpa indexes repo-local skill dirs only, so agent-reach never appears in its output. |
| 4 | **Local patches get destroyed** | `_install_skill(force=True)` `rmtree`s and rewrites the skill dir on **every** `agent-reach install` / `skill --install`. Unlike `skills-lock.json` skills (pinned commit + hash + recorded `localPatch`), nothing here is pinned or preserved | Treat as vendor-owned and never patch, **or** vendor it into the repo under `skills-lock.json` | **VENDOR-OWNED, NEVER PATCH.** Recorded as an explicit exception in `skills-lock.json` (`policy` field). To change behaviour we change how *we* call it, never its files. It is deliberately NOT vendored into `.agents/skills/`. |
| 5 | **Wrong language installed** | With `LANG`/`LC_ALL` unset the installer picks the Chinese `SKILL.md`; `SKILL_en.md` exists. Fix is one command: `AGENT_REACH_LANG=en agent-reach skill --install` | Swap to English, or leave | **SWAP TO ENGLISH — done.** Reinstalled 2026-08-24 via `AGENT_REACH_LANG=en agent-reach skill --install`; both skill roots now carry the English `SKILL.md` (v1.5.0) and the 7 English `references/*.md`. |
| 6 | **Not in `skills-lock.json`** | Provenance gap: an unpinned, auto-overwriting, user-level skill in a repo whose convention is pinned+hashed vendoring | Record it, or accept the exception | **RECORD IT — done.** `skills-lock.json` now has an `agent-reach` entry: source archive + ref, v1.5.0, install method (`uv tool`, pipx absent), the machine-wide install locations, `skillLang: en`, a tree hash for **drift detection only** (it is not a pin — the installer rewrites the tree), `localPatch: null`, and the never-patch `policy` text. |

## Decision and next step

**Decision:** agent-reach is **worth keeping installed** for exactly two channels — Jina primary-source fetch (adopt, narrow) and event-triggered YouTube transcripts (adopt, narrow) — with Exa parked as a discovery aid and everything else parked or out of scope. It is not a new pipeline; it is three shell commands plus a health check, and every durable property (timestamps, caching, display flags, weights) stays ours.

**Nothing was wired.** `run_daily.py`, `feeds.py` and `dashboard.py` are untouched; no cookies, no logins, no account. The only artifacts of this pass are the machine-level installs recorded in § Install record and this note.

**Next step, in order:**

1. ~~**Justin answers the three sign-off blocks** above (weights, burner-Twitter, skill conflicts).~~ **DONE 2026-08-24** — weights APPROVED as proposed (and the v3 "unknown" they inherit is now 0.5, live); official primary documents may display headline + source + link with bodies internal; reach slot ≤2 displacing Guardian 6→4 approved; burner-Twitter **NO**; skill housekeeping executed (English reinstall + `skills-lock.json` never-patch entry, kept machine-wide). **The reach weights remain inert because nothing is wired** — approval is not a build.
2. **If adopted:** a one-time curation pass to pin the official resolution-source URL(s) per market — the honest bulk of the work, and the highest-value part. Start with the **Hormuz** family, which is both the thinnest currently covered and the one where the right URL is genuinely unknown.
3. **Then** the build slice: reach worklist in `--stage fetch`, `reach_items.json` out-of-band ingestion, the two-level fragility gate, the chrome-strip, the display filter at the page boundary, and the tests listed in § Display rules. Small, additive, and matching the existing out-of-band idiom.
4. **Do not** claim any forecast improvement until settled forecasts have been refit through `scripts/newsagent_hist_backfill.py --fit` under the extension. Reach is not edge.

---

# Addendum — 2026-08-24: what shipped vs what was designed (BUILD SLICE)

> Written the same day as the sign-off above, immediately after the build. Everything before this line is the **scoping** pass and its recorded decisions; everything below is the **build**. Where the two disagree, this addendum is the record of what is actually in the repo.
> Companion: [[newsagent_observatory_v33_findings]] (the v3.3 state this builds on) · [[CODEX]] § Realism calibration.

## Plain-English Summary

- **What shipped.** The reach extension is wired and live: a curated per-market worklist emitted by `--stage fetch`, an out-of-band `--reach-file` ingestion path, a two-level fragility gate (doctor pre-flight + per-item validation), a mandatory chrome-strip, the approved display rules enforced at the page boundary, and the approved source-weight rows as declared constants. 60 new tests, **613 green repo-wide** (553 before), one full attended cycle run with the extension live.
- **The finding that matters, and it is a negative one.** The design's load-bearing mechanism — *date a Jina item by its `Published Time` header* — was verified on two federalreserve.gov pages during scoping and **does not generalise**. Probing 22 curated official sources on 2026-08-24: **exactly one** (federalreserve.gov dated press releases) returns a usable publication timestamp. The scoping note's own assumption ledger flagged this risk ("it is Jina's documented behaviour, but it is two data points"); the build measured it and the assumption failed.
- **A worse failure mode than "no timestamp", discovered during the build.** centcom.mil served `Published Time: Mon, 24 Aug 2026 13:28:03 GMT` for an article datelined **July 29, 2026** — the fetch instant, 26 days late. A fetch-clock stamp is never *after* the cutoff, so the approved after-cutoff check cannot catch it, and it would have entered a month-old document into the packet as today's breaking evidence. nobelprize.org does the same. This is now refused in code.
- **The inversion.** Scoping expected Jina to be "the main find" and video to be the narrow adjunct. Empirically it is the other way round: the page channel reaches **one** domain with a defensible date, while yt-dlp carries real upload timestamps on **every** channel tested. The transcript path is the one that generalises.
- **What today's live cycle actually contributed: nothing.** One item validated (an 8,025-word State Department transcript), and the declared 72h transcript window correctly excluded it for being 32 days old. Three submissions were rejected with three distinct real reasons. **Zero reach items reached any packet**, so no published number moved because of this build, α was not refit, and there is no forecast claim of any kind here. Reach is not edge, and today it was not even evidence.

## 1 · What shipped, item by item, against the design

| Designed (§ Integration / § Display / § Fragility) | Shipped | Deviation |
|---|---|---|
| Reach worklist emitted from `--stage fetch` | `newsagent/reach.py::build_worklist` + `reach_worklist.json` in the day dir; 20 jobs today | none |
| Curated per-market official URLs + event triggers, as config data | `newsagent/reach_sources.json` — 19 markets with sources, 3 with a recorded reason for having none, 4 trigger kinds, 3 declared calendars | none |
| `--stage fetch --reach-file reach_items.json` merging article-shaped dicts through `_keyword_filter` → `build_packet` | shipped | **the agent submits RAW payloads, not article-shaped dicts** (§ 2) |
| Windows: 14d official documents / 72h transcripts, declared | `reach.OFFICIAL_DOC_WINDOW_DAYS = 14`, `TRANSCRIPT_WINDOW_HOURS = 72`, applied per kind in `build_packet` | none |
| Stage-A cache-key collision fixed (content-unique titles unless immutable-dated) | `reach.content_unique_title` — immutable documents keep their headline verbatim (they are displayed), everything else gets a `date · sha1[:8]` suffix | none |
| Doctor pre-flight cached to the day dir; missing binary = clean skip | `reach.doctor()` → `reach_doctor.json`, 5s timeout; `shutil.which` miss returns `available: False` | none |
| Per-item validation: Jina 4xx-in-JSON / missing Title / 404-warning / min-length post-strip | `reach.parse_jina`, all four, in that order | **plus a fifth check** (§ 3) |
| yt-dlp ≥200 words post-cue-strip | `reach.strip_vtt` + `MIN_TRANSCRIPT_WORDS = 200`; consecutive-duplicate collapse added so a rolling auto-caption track cannot inflate its way over the floor | none |
| Failures ABSENT, never placeholders | `ingest_reach_file` returns `(items, rejected)`; rejections are logged to `reach_rejected.json` and nothing is written for them | none |
| Chrome-strip mandatory | `reach.strip_chrome`, two passes; measured **52,119 → 1,462 bytes (2.8%)** on the July FOMC statement with the decision sentence intact | none |
| Page-boundary display filter + test obligations | boundary generalised from "domain starts with `newsletter:`" to "**`display=False`, belt-and-braces re-checked against never-public domain prefixes**", enforced in all three places a title can escape (§ 4) | strengthened |
| Official docs render headline + source + link | `display=True`, body internal-only (`scan_text`) — the `pdf_ingest` treatment | none |
| New weight rows as declared constants, blocklist precedence preserved and test-enforced | `sourceweights.OFFICIAL_PRIMARY_W` 1.0 / `OFFICIAL_STATE_CLAIM_W` 0.5 / `OFFICIAL_TRANSCRIPT_W` 1.0 / `UNKNOWN_VIDEO_W` 0.5 + domain sets | none |
| Reach ≤2 displacing Guardian 6→4 | `MAX_REACH_ITEMS = 2`, `GUARDIAN_CAP_WITH_REACH = 4`, applied only on days a reach item survives | none |
| Timestamp rules verbatim, incl. `max(upload, release+duration)` and the rolling-page discard | shipped verbatim | **plus a tightening** (§ 3) |
| Exa discovery-only, never an evidence row | not wired at all this pass — no code path can turn an Exa result into an item | narrower than designed |
| No Twitter / no social channels | nothing installed, no browser profile read | none |

### 1a · Three deviations, all declared

**(i) The agent hands over bytes, not items.** The design said the attended agent "applies the validation gate and writes article-shaped dicts". Shipped the other way round: the agent writes the raw `r.jina.ai` response and the raw `yt-dlp` output, and **the gate runs in the repo**. Reason: a gate that lives in the agent is a gate that cannot be tested, cannot be regression-protected, and silently changes whenever the session model changes. `source_type`, `domain`, `display` and `immutable` are taken from the curated worklist, never from the payload, so a submission cannot promote itself into a heavier weight row or a different domain. This is strictly stronger than the design and is what the 60 tests actually test.

**(ii) `index` vs `document` roles.** The design treated "rolling page" as a discard condition. Shipped as a first-class role: a rolling page is `role: "index"` — **navigation only**, fetched so the attended agent can find today's dated document, and `item_from_jina` refuses to build an item from one. Index entries may declare a `child` block pinning a `url_prefix` (pages) or a `channel` (video); a discovered submission must sit under it. Without this the curation would have been unusable, because most official sites publish at unpredictable per-article IDs (`centcom.mil/…/Article/4559495/…`) that no URL template can reach.

**(iii) A fifth Jina check and a timestamp-trust gate.** Both are *tightenings* — they can only ever make an item absent. See § 3.

## 2 · The curation pass — what the official web actually gives us

The scoping note called curation "the honest bulk of the work, and the highest-value part". It was. 22 web sources were probed live through the Jina path on 2026-08-24, plus 4 YouTube channels through yt-dlp.

**Column meanings.** *Timestamp class* is what the `Published Time:` header did, and it is the only column that decides whether a source can produce evidence: `verified` = equals the document's release instant; `touch` = a real edit time on a rolling page; `fetch_clock` = tracks the fetch instant, so it is always "now"; `absent` = no header; `blocked` = r.jina.ai could not reach the origin. *Shipped as* is the role in `reach_sources.json`.

| Source | Timestamp class | Shipped as | Note |
|---|---|---|---|
| federalreserve.gov dated press release | **verified** | **document** ✅ | `Wed, 29 Jul 2026 18:00:15 GMT` = the 2:00pm ET release instant |
| federalreserve.gov `fomccalendars.htm` | touch | index | `19 Aug 2026` = the July-minutes edit, not an event |
| sos.ca.gov ballot measures | touch | index | `23 Jun 2026`, a genuine edit on a rolling index |
| centcom.mil article | **fetch_clock** | index, child disabled | **26 days late** — see § 3 |
| nobelprize.org peace prize | **fetch_clock** | index | `Last-Modified` == the fetch instant |
| state.gov `/releases/…/2026/08/<slug>/` | absent | index, child disabled | body datelined **20 Aug 2026**, origin `Last-Modified: 17 Mar 2026` — five months stale |
| state.gov press releases / press briefings | absent | index | indexes work and list dated URLs |
| congress.gov H.R.3633, `/all-actions`, House resolutions | absent | index | no `Last-Modified`, no published-time metadata |
| ukmto.org recent-incidents / warnings / advisories | absent | index | **richest content found** — see § 2a |
| imo.org, nato.int, war.gov, whitehouse.gov, gov.il, tse.jus.br, conseil-constitutionnel.fr, kremlin.ru | absent | index | all reachable, all undated |
| iaea.org | **blocked** | disabled | Cloudflare 403 "Just a moment…" |
| president.gov.ua | **blocked** | disabled | 403 Access Denied (430-byte body) |
| interieur.gouv.fr | **blocked** | disabled | Cloudflare "Attention Required!" 403 |
| ukmto.org weekly VRA reports | absent (JS-only) | disabled | listing renders **empty** — client-side JS the keyless Reader does not execute |
| cec.gov.ru | blocked | disabled | 242-byte empty shell |
| YouTube @federalreserve / @statedept / @centcom / @nobelprize | **verified** | video | yt-dlp carries real upload timestamps on all four |

**Read.** **1 of 22 web sources can date a document; 4 of 4 video channels can.** The mechanism is not mysterious: federalreserve.gov serves static `.htm` files with a true `Last-Modified`, and essentially every other modern government site is CDN-fronted and sends either nothing or a cache timestamp. This is a property of how governments host, not of Jina, and no amount of curation fixes it.

**Consequence, stated plainly.** The Jina channel as approved delivers evidence for the **Fed family and nothing else**. Every other curated source ships as navigation. That is a much narrower channel than "ADOPT — narrow" was understood to mean at sign-off, and it is the single most important thing on this page.

### 2a · The Hormuz find, and why it does not count

The curation started with Hormuz as instructed, and it worked: `ukmto.org/recent-incidents` carries dated per-incident narratives — several naming the Strait of Hormuz directly — from the last fortnight. Verbatim shape: *"UKMTO WARNING 118-26 - HIJACK Incident Date: 20 Aug 2026 … a tanker has broadcast a distress call"*; *"18/08/2026 ATTACK: … the vessel was struck by an unknown projectile while conducting an outbound transit of the Strait of Hormuz"*. Nothing in the current stack sees any of this: Guardian covers escalation, it does not report the operational record.

It carries **no `Published Time`**. Under the approved rule it is navigation only and cannot become a dated evidence item. The dates are printed in the body text — which is precisely the amendment proposed in § 6, and precisely why that amendment should be decided rather than assumed.

Also worth recording: the UKMTO **Weekly VRA Overview Report** would be the single best instrument for a "traffic returns to normal" question — a weekly official summary of exactly the thing being asked. Its listing page renders empty through the keyless Jina tier because the list is client-side JavaScript. Recorded so the next pass does not rediscover it.

### 2b · Curation coverage

19 of 24 live markets have curated sources. The other five are recorded in `markets_deliberately_without_sources` with reasons: the two midterm-control markets (the note's own mapping table says Politico/The Hill already cover this densely, and no official document bears on a seat-count forecast before election night) and Becerra (`POLL_DRIVEN` — no official statistic tracks it; adding an official page would look like coverage without being it). A test enforces that every live market is in one bucket or the other.

## 3 · Two tightenings added during the build

Both can **only ever make an item absent**. Neither introduces a new timestamp source, and neither admits anything the approved rule would have rejected. A *loosening* would need sign-off before it binds; a tightening does not, and both are recorded here rather than in a commit message.

**(i) The fetch-clock class.** Measured live: centcom.mil returned `Published Time: Mon, 24 Aug 2026 13:28:03 GMT` for an article whose own body is datelined **July 29, 2026**. The drift is 26 days *in the direction that makes stale content look like breaking news*, and because such a stamp is always ≈now it is never "after the cutoff" — the approved check cannot see it. Shipped guard: a document is only ingestable if its curated `probe.published_time` is `verified`, enforced in `item_from_jina` via a `timestamp_trusted` flag that `build_worklist` derives from the recorded probe. A config edit alone cannot promote a source into the trusted class. Tests: `test_a_fetch_clock_timestamp_is_refused`, `test_an_unmeasured_source_cannot_date_evidence`, `test_only_verified_sources_carry_a_trusted_timestamp_in_the_worklist`, `test_the_data_file_never_claims_verified_without_a_probe`.

**(ii) A minimum-prose floor after chrome-strip.** `MIN_DOC_CHARS = 400`. A page that renders to almost nothing after navigation removal is an interstitial, a stub or a failed render, not a document.

## 4 · The display boundary, generalised

The `email_ingest` pattern was "items whose domain starts with `newsletter:` are private". Transcripts are `display=False` on **every** source, so a prefix check was no longer sufficient. The boundary now keys off the `display` flag **and** re-checks the domain against `NEVER_PUBLIC_PREFIXES = ("newsletter:", "youtube.com/@")` — belt and braces, because a privacy boundary must not depend on an upstream field being present.

Three places a title could reach the public, all now filtered (v3.1 caught a latent leak precisely because the filter existed at the source but not at the page):

1. `dashboard.build_showcase` → card evidence lists and the shared feed pane.
2. `dashboard._bd_row_public` → the FV-construction breakdown table. `fvmodel.breakdown` now carries `display` on each row so the page can honour it.
3. `run_daily.stage_publish` → the `drivers` list, which is public copy **and is written to the append-only ledger**. This one previously keyed off the `newsletter:` prefix and would have leaked a transcript title — including its content-hash cache suffix — into a book that refuses edits.

Test obligations from § Display rules, all discharged: a `display=False` reach item never appears in the feed pane or a card's evidence list; `n_private` increments; the IP-scrub end-to-end test passes with reach items present; **a transcript is unreachable from the page regardless of its domain**; and a legacy breakdown row with no `display` key still hides a transcript.

Mutation-checked rather than assumed: removing the undated-discard rule fails 1 test; weakening the display boundary back to the `newsletter:` prefix fails 3.

## 5 · The live cycle, 2026-08-24

A full attended run with the extension live: `fetch` → reach worklist → attended fetches → `fetch --reach-file` → `extract` → `publish` → ledger → dashboard.

| Stage | Result |
|---|---|
| Doctor pre-flight | `available: true`; `web` ok (Jina Reader), `youtube` ok (yt-dlp), `exa_search` warn — cached to `reach_doctor.json` |
| Worklist | **20 jobs** (19 index, 1 video discovery), deduped across markets — one state.gov fetch serves 5 markets, Kremlin 4, CENTCOM 3, NATO 3. **6 sources skipped** as disabled-by-probe, 0 channel-skipped |
| Triggers | Correct: no FOMC jobs (2026-08-24 is not an FOMC day), no Nobel jobs (outside the 05–15 Oct window), State Dept briefing job fired (Monday) |
| Attended fetches | 4 real payloads submitted |
| Gate | **1 validated, 3 rejected** |
| Reach items reaching packets | **0** |
| Guardian | now live on the real key (`GUARDIAN_API_KEY` pasted into the git-ignored `.env`), which surfaced **9 new articles** the demo key had not |
| Stage A | 9 extractions, out-of-band in-session, `_meta.source` stamped `oob:attended-2026-08-24b` — α remains Haiku-era, so the v3.3 provider caveat is unchanged |
| Publish | 24 markets, 3 divergence flags (Fed-Sep −22.5pp, Dems-Senate −17.8pp, Russia–Ukraine ceasefire −15.9pp) — the same three as v3.3 |
| Tests | **613 green** (553 before this build; +60 new) |

**The four submissions and what the gate did with them** — this table is the per-item validation working on real objects, which is the only way to know it works:

| Submitted | Verdict | Reason |
|---|---|---|
| State Dept official-channel transcript (`eRRL8oLAjwM`, Rubio remarks to the press) | **validated** | channel `@statedept` matched the curated channel; `live_status: not_live`; dated `2026-07-23T14:45:36Z` by `max(upload, release+duration)`; 8,025 words after cue-strip; `display=False` |
| state.gov Iran-sanctions release (real, datelined 20 Aug) | rejected | documents under this index are disabled — origin `Last-Modified` is five months stale |
| centcom.mil IRGC-strike article (real, datelined 29 Jul) | rejected | documents under this index are disabled — article pages return a fetch-clock timestamp |
| `ukmto.org/recent-incidents` submitted **as an item** | rejected | *"rolling/index page — its Last-Modified is a touch time, not an event time; index pages are navigation only, never items"* |

**And then the one survivor contributed nothing.** The transcript is dated 2026-07-23 and the declared transcript window is 72h, so `build_packet` excluded it for being 32 days old. `n_reach = 0` on all 24 markets. That is the design working exactly as specified — a valid object correctly excluded by a declared window — and it means **no published number on 2026-08-24 moved because of this build**.

**Why the State Dept channel produced nothing usable.** Its recent uploads are Secretary-level "Remarks to the Press" from 20–23 July and evergreen history shorts; the curated `match` string is "Department Press Briefing", and in any case everything on the channel is well outside 72h. The `match` string was **not** widened to manufacture an item — the window would have excluded the result either way, and widening curation to make a demo produce output is fitting config to outcome.

### 5a · An unrelated live finding worth recording

`kremlin.ru` is **on the Iffy blocklist**, so it resolves to weight **0.0**, not the approved 0.5 `official_state_claim` row. That is the approved precedence working as designed — a blocklisted domain cannot be resurrected by a declared row — and it is now a live example rather than a hypothetical. Test: `test_blocklist_beats_an_official_reach_domain`. Practical effect: even if the timestamp problem is solved, Kremlin readouts carry zero Stage-B weight; they would be fetched and counted, never influential.

### 5b · Two agent-reach gotchas, both silent

- **`yt-dlp --dump-json` implies simulate** and writes **no** subtitle file, while printing metadata that looks like success. `--print-json` prints the same metadata *and* writes the files. The scoping note's recorded command has the former.
- **`--sub-lang en` (singular) matched nothing** and exited 1 without a file; `--sub-langs "en.*"` is what selects the English track on this yt-dlp build. Both are now baked into `reach._ytdlp_command` with a test (`test_worklist_yt_command_actually_writes_a_vtt`).
- Minor: `agent-reach doctor --json` returns its prose `message` fields **in Chinese** on this machine despite the English skill reinstall — the CLI localises by `LANG`, and the skill's language is a separate thing. The gate reads `status`/`active_backend`, which are language-independent, and quotes the message verbatim.

## 6 · Proposed amendment — NOT implemented, needs sign-off

**The problem.** As shipped, the page channel reaches one domain. Every other curated official document — state.gov readouts, CENTCOM releases, UKMTO advisories, congress.gov actions — is an immutable dated document that *states its own date in its body and often in its URL path*, and is discarded because the HTTP transport does not carry that date.

**The proposal.** Add a **second declared timestamp source**, used only when `Published Time` is absent or untrusted, and only for sources whose entry declares it:

1. the publication date embedded in the URL path (`/releases/office-of-the-spokesperson/2026/08/<slug>/` → 2026-08); **and**
2. a dateline parsed from the document body against a per-source declared pattern (`"August 20, 2026"`, `"Incident Date: 20 Aug 2026"`),

with the item **discarded on disagreement**, discarded if either is missing, and dated conservatively at 23:59:59Z of the stated day (the same convention the approved rule already gives a date-only Exa result).

**Why it is not in this build.** It is a *loosening* — it admits items the approved rule rejects — and the approved timestamp discipline is the thing standing between us and a lookahead violation. The instruction for this pass was the rules verbatim. It should be decided on its merits, with the evidence above, not adopted because the channel would otherwise be thin.

**Honest argument against it, so the decision is a real one.** A dateline is *content*, and content is exactly what an adversarial or sloppy source controls. The `Last-Modified` path is at least an infrastructure fact. If the dateline rule is adopted, the first thing to test is whether a document's body date can ever be later than the fetch that found it — i.e. whether it can manufacture a future-dated item — and the `discard on disagreement` clause exists for that.

**Decision needed from Justin:** adopt the two-source dateline rule as described, adopt a narrower variant (URL-path only, which is machine-structural rather than content), or hold the line and accept that the page channel is a Fed-only path.

## 6a · AMENDMENT — the two-source dateline rule, LOCKED 2026-08-24 (Justin's variant)

> **DECISION — Justin, 2026-08-24: ADOPT the two-source dateline rule, with a TIE-BREAK instead of discard-on-disagreement.** Locked here **before** any implementation and before any source was re-probed under it. This is a **loosening** of the approved timestamp discipline — it admits items the § Timestamp-discipline rule discards — which is exactly why it is written down as a dated amendment rather than slipped in as a "tightening". Everything not restated below is unchanged: `Published Time` remains the primary path, `verified` sources are untouched, rolling pages stay `role: index`, and failures remain ABSENT rather than degraded.

**DL-1 · When the second path may fire.** A source entry may declare a `dateline` block. It is consulted **only** when the primary path cannot date the item — i.e. `Published Time` is **absent**, or present but classed **untrusted** (`fetch_clock` or `touch`). A source whose probe class is `verified` never reaches this code, and a source with **no declared `dateline` block behaves exactly as it did before this amendment**.

**DL-2 · The two sources, both declared per source.**
1. **URL-path date** — a declared regex over the document URL yielding a year and a month, optionally a day (`state.gov/releases/office-of-the-spokesperson/2026/08/<slug>/` → 2026-08). This leg is **machine-structural**: it is part of the address the publisher minted, not text the page can rewrite.
2. **Body dateline** — a declared regex over the chrome-stripped body yielding a full calendar day (`"August 20, 2026"`, `"Incident Date: 20 Aug 2026"`).

**DL-3 · Agreement → date the item at the body dateline.** "Agree" means the URL date **contains** the body day, checked at the URL's own precision: a month-precision URL agrees when the body day falls in that year-month; a day-precision URL agrees on an exact match. The item is dated at the body day, **23:59:59Z** — the existing conservative convention, the same one the approved rule already gives a date-only Exa result.

**DL-4 · Disagreement → tie-break, NOT discard.** The item survives and is dated at the bound that cannot flatter us:
* **Live packets → the EARLIER of the two.** A stale document must never present as fresh; this is the centcom.mil failure mode (a July article stamped with today's clock) written into the rule rather than left to a guard.
* **Historical reconstruction → the LATER of the two.** Nothing may enter a packet dated before it existed; this is the GDELT/`release_timestamp` discipline, unchanged in spirit.

Because the URL leg may be month-precision, "the two" are made comparable by taking the extreme instant consistent with each source: the earlier branch uses `min(body_day, first day of the URL month)`, the later branch uses `max(body_day, last day of the URL month)`, both at 23:59:59Z. For a day-precision URL the two collapse to that day.

Every tie-broken item carries **`dateline_conflict: true`**, so the conflict rate is countable rather than invisible. A rising rate on a source is the signal to demote it back to `index`.

**DL-5 · Either leg missing → DISCARD, at this stage.** No single-source dating: a body dateline with no URL date, or a URL date with no parseable dateline, is not enough. The reason is recorded per item so the miss rate is measurable. **Context-based inference — dating a document from an index page's listing, from a neighbouring item, or from the position of an anchor on a rolling page — is explicitly DEFERRED, not rejected.** It is the obvious next amendment and it needs its own decision.

**DL-6 · Hard refusal, non-negotiable.** Any resolved date **later than the fetch instant** is discarded as future-dated, with its own distinct reason, whatever the two legs say. The fetch instant is the run cutoff (in an attended run these coincide to within the session), narrowed by a record's own `fetched_at` when supplied. This is the guard against the one thing a *content*-derived date can do that an infrastructure-derived one cannot: manufacture a document from the future. It is tested adversarially, not just asserted.

**Precedence, restated end to end.** Iffy blocklist → `Published Time` when `verified` → the dateline path when a `dateline` block is declared and the primary path is absent/untrusted → otherwise ABSENT. The display rules, the source weights, the packet budget and the windows are all unchanged by this amendment.

**The honest argument against, restated because adopting it does not make it go away.** A dateline is **content**, and content is what an adversarial or sloppy source controls; `Last-Modified` is at least an infrastructure fact. The mitigations are that both legs must agree or be tie-broken conservatively, that the URL leg is structural rather than textual, that a future date is refused outright, and that the conflict rate is counted. None of that makes a dateline as trustworthy as a release instant, and no item dated this way should ever be described as carrying one.

---

## 6b · What the amendment unlocked — implemented and measured, 2026-08-24

> Built immediately after § 6a was locked, in that order. **Headline: the amendment unlocks exactly ONE source family — `state.gov/releases/` — and it does not unlock its own priority-1 target.** The reach channel went from **0 items reaching packets to 2**, both the same official document, into the two Iran markets.

### The re-probe, source by source

DL-2 requires **both** legs. The body-dateline leg is easy — official documents print their date. The binding constraint is the **URL-path leg**, which is the leg deliberately kept because it is structural rather than content. Every priority source was re-probed live under the amendment before being enabled:

| Source (priority order) | Body leg | **URL leg** | Verdict |
|---|---|---|---|
| **ukmto.org** recent-incidents / warnings / advisories | **yes** — the text is full of them (`Incident Date: 20 Aug 2026`, `18/08/2026 ATTACK:`, `Report Date: 08 Aug 2026`) | **NONE, anywhere on the site** | **stays `index`** |
| **state.gov** `/releases/…/2026/08/<slug>/` | yes — `August 20, 2026`, 322 chars into the stripped body, directly under the headline | **yes** — `/2026/08/` | **ENABLED** ✅ |
| **centcom.mil** `/MEDIA/PUBLIC-RELEASES/Article/<id>/<slug>/` | yes — `July 29, 2026` | **NONE** — the path carries an opaque article id | stays `index` |
| **congress.gov** `/bill/119th-congress/house-bill/3633/all-actions` | n/a | **NONE** — and the page is rolling anyway | stays `index` |

**The UKMTO result is the disappointing one and it deserves the detail, because it was the priority-1 target and the best instrument on the slate for the two Hormuz markets.** It was checked exhaustively, not glanced at: `/recent-incidents` is a single rolling page whose per-incident anchors are opaque GUIDs (`#dc41f5ed-d55c-43db-9bc8-9d45bc7604cd`); `/ukmto-products/warnings` drills down only to **year**-level URLs (`/warnings/2026`), and that page renders a JavaScript month accordion showing counts (`March (36)`, `July (25)`) with no per-warning links the keyless Reader can follow; `/ukmto-products/advisories` has no drill-down at all. **There is no address anywhere on ukmto.org that carries a month.** Under DL-5 a body dateline alone is not enough, so the richest evidence found in the entire curation pass remains navigation-only.

Dating a UKMTO incident would require inferring its date from its position on a rolling page or from an index listing — which is exactly the **context-based inference DL-5 explicitly defers**. The amendment anticipated this category; it just turned out to contain the one source we most wanted.

### The live cycle, 2026-08-24 (second attended run of the day)

| | Result |
|---|---|
| Worklist | 20 jobs, unchanged |
| Submitted | 7 raw payloads (1 transcript, 4 state.gov documents, 1 CENTCOM article, 1 UKMTO index-as-item) |
| **Validated** | **3** — the State Dept transcript + **2 state.gov documents, both dated by the two-source rule** |
| Rejected | 4, with four distinct real reasons (below) |
| **Reach items reaching packets** | **2** — up from **0** in the previous build |
| Dateline conflicts | **0 of 2** tie-broken — both documents' legs agreed |
| Tests | **636 green** (613 before; +23 for the amendment) |

**The two documents that landed**, and where:

| Document | Dated | How | Markets it reached |
|---|---|---|---|
| *U.S. Sanctions Smuggling Network for Qods Force and Hizballah* | 2026-08-20 23:59:59Z | DL-3 agreement (`/2026/08/` ∋ `August 20, 2026`) | **Iranian regime falls**, **US invades Iran** |
| *United States Participates in Third APEC 2026 Senior Officials' Meetings in Dalian, China* | 2026-08-22 23:59:59Z | DL-3 agreement | none — validated, then correctly dropped by the per-market keyword filter (the China–Taiwan market keys on `taiwan`, which the release does not mention) |

On both Iran markets the packet's Guardian slot dropped **6 → 4** as approved, so the document displaced two restatements rather than growing the packet. On the public page the release renders as **headline + source + link** with the body internal-only — the `pdf_ingest` treatment — verified by inspecting the built page: the headline and the `state.gov/releases/…/2026/08/` link are present, the body sentence is not.

**The four rejections**, each a different rule doing its job:

| Submitted | Reason |
|---|---|
| state.gov *Ukraine National Day* (datelined **today**) | **DL-6 future-dated** — resolves to 2026-08-24 23:59:59Z, later than the 20:32Z fetch instant |
| state.gov *Rescission of Syria's SST designation* (datelined **today**) | same |
| centcom.mil IRGC-strike article | child disabled — article pages return a fetch-clock timestamp, and there is no URL leg to rescue them |
| `ukmto.org/recent-incidents` submitted as an item | rolling/index page — navigation only, never an item |

### The measured consequence nobody specified: the dateline path is a T+1 channel

**Two of the four state.gov documents were refused for being published *today*.** This is not a bug in DL-6 and it is not new to the amendment — it falls out of two rules that were already in force. DL-3's conservative convention dates a day-precision document at **23:59:59Z**, and a live run's cutoff is `min(end-of-day, now)`. A document published on the run date therefore always resolves *after* the cutoff, and would have been refused by the pre-existing after-cutoff check even if DL-6 had never been written; DL-6 simply reaches it first and says so more legibly.

The practical cost is real and should be stated rather than discovered later: **the freshest documents — today's readout, today's statement — are exactly the ones this channel cannot ingest.** It can only carry documents from previous days. On a slate of shock-type geopolitics markets that is the wrong day to be blind on.

The obvious repair is to **clamp a same-day document's stamp to the fetch instant instead of discarding it**, which is provably not a lookahead — we fetched the document, so it existed by then — and is strictly *more* conservative than 23:59:59Z. It is nevertheless a rule change that converts a discard into an admission, so it is **proposed, not taken**: it needs its own dated amendment, the same way this one did.

### Chrome-strip repairs found by running it

Two site-boilerplate classes survived every existing rule because they are *prose*, and both were caught only by looking at what Stage A actually received:

- **state.gov's cookie-consent block** — ~3 KB of consent text sitting *above* the document's own dateline. It pushed the dateline to char 3004 and would have been the first thing the extractor read.
- **state.gov's global consular alert bar** (`Americans in Colombia needing consular assistance can call…`), emitted **twice**, which `prose_paragraphs` was handing Stage A as the item's `lede`. The first extraction attempt of the day read a travel advisory instead of a sanctions release.

Both are now declared line patterns in `BOILERPLATE_PATTERNS`, joined by a consecutive-duplicate collapse. After the fix the lede is the document's opening sentence. This is the `pdf_ingest` disclaimer-strip precedent doing its job one site at a time, and it is a standing maintenance cost of the page channel: **every new domain brings its own furniture.**

### What moved on the page, and what may not be claimed about it

The two Iran markets moved: *Iranian regime falls* **8.7% → 17.4%**, *US invades Iran* **6.7% → 4.5%**. **This is not attributable to the reach document and is not offered as evidence of anything.** Three things changed in those packets in the same run: the official document arrived, the Guardian feed turned over and delivered a new sanctions story to both markets, and the Guardian slot shrank 6 → 4. The contributions are not separable and no attempt was made to separate them. α was **not** refit and stays 2.85 — the fit sample is historical and contains no reach items, so no refit is due. **Reach is not edge; this section reports plumbing, not performance.**

---

## 7 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** That `probe.published_time` measured once on 2026-08-24 stays true — it is one observation per source, and centcom.mil's class *changed between two probes hours apart in this very session* (absent on the HEAD sweep, fetch-clock on the Jina fetch), so this is known-unstable and the probe date is recorded on every entry for exactly that reason. That the chrome-strip's declared thresholds (`PROSE_WORDS_MIN = 12`, `ANCHOR_GAP = 1`, `NAV_ITEM_MAX = 120`) generalise beyond the pages tested — they were checked against the FOMC statement and a handful of index pages, not tuned against any outcome, and a document that is genuinely a short bulleted advisory will lose its bullets. That reach ≤2 displacing Guardian 6→4 is a net evidence gain: **still plausible, still unmeasured, and now untestable for a different reason — it has never once bound**, because no reach item has reached a packet. That the declared weight rows are judgments, not fits.

**Live-only unknowns.** Whether the Fed path survives contact with a real FOMC day — the September 16 meeting is the first live test of the calendar trigger, the statement fetch and the presser transcript together, and until then the one working document path is **unexercised in production**. How often the keyless Jina tier blocks a domain we need, and for how long (three blocks observed in one session; no base rate). Whether official-channel captions stay human-written rather than degrading to ASR. Whether the worklist stays maintainable as the slate refreshes — 19 markets of curation is now a maintenance surface that did not exist yesterday. And the only question that matters, unchanged and unanswerable offline: **whether primary-source evidence improves forward Brier**, which needs settled forecasts under the extension refit through `scripts/newsagent_hist_backfill.py --fit`.

**Power honesty.** Nothing here is a measurement of forecast quality, and this build produced **no forecast-relevant change at all**: zero reach items entered any packet, α was not refit and did not move, and the three divergence flags on the 2026-08-24 page are the same three as v3.3. The [[newsagent_v0_gate_findings]] closure is untouched and stays displayed. **Reach is not edge** — and on the evidence of this build, reach is not yet even reach for 21 of the 22 sources curated.

## 8 · Outputs

- **New code:** `newsagent/reach.py` (the worklist, the gate, the strips, the timestamp rules), `newsagent/reach_sources.json` (the curation).
- **Modified:** `newsagent/feeds.py` (reach slot + Guardian displacement), `newsagent/run_daily.py` (`--reach-file`, `_reach_pass`, driver-line boundary), `newsagent/sourceweights.py` (four declared rows + domain sets), `newsagent/fvmodel.py` (`display` on breakdown rows, transcript source label), `newsagent/dashboard.py` (page boundary, transcript/doc counts).
- **Tests:** `tests/test_newsagent_reach.py` — 60 tests. Repo-wide **613 green** (553 before).
- **Day state** (git-ignored): `data/newsagent/live/2026-08-24/` gains `reach_doctor.json`, `reach_worklist.json`, `reach_items.json`, `reach_items_validated.json`, `reach_rejected.json`, `reach_raw/`.
- **Credential:** `GUARDIAN_API_KEY` value pasted into the git-ignored `polymarket/research/.env`; verified live (5/5 rows on a probe query, 9 new articles in the run). The demo-key fallback in `feeds.py` is now only reached if that line is emptied.
- **Not touched:** α (`fv_params.json` unchanged at 2.85), the fit sample, any deployment. The Vercel push is Justin's.
- **Still open from v3.3:** Gmail re-consent (newsletters were absent from this run too), `GEMINI_API_KEY` for the provider spot-check, `ANTHROPIC_API_KEY` for the unattended switch.

## 9 · Decision and next step

**Decision: the extension is built, tested and live, and it currently reaches almost nothing.** The gate, the display rules, the weights and the packet budget all work and are test-enforced on real payloads. The curation is done and is the durable asset. What does not work is the assumption underneath the whole page channel — that an official document's HTTP transport carries its publication date — and that is now measured rather than assumed.

**Next, in order:**

1. **Justin decides § 6** (the dateline amendment). This is the difference between a Fed-only channel and the channel the mapping table describes. Nothing else on this list changes that.
2. **September 16** is the first real test: the FOMC calendar trigger, the statement document and the presser transcript all fire together for the first time. Watch the run log, not the number.
3. **Do not refit α** until a reach item has actually entered a packet. Refitting on a composition change that never happened would be a false attribution — the v3.3 α move is attributable precisely because its cause was isolated.
4. **Re-probe the curated sources periodically.** `probe.published_time` is one observation per source and one of them already moved class inside a single session. A cheap `scripts/` re-probe that diffs against `reach_sources.json` is the obvious follow-up, and it is not built.
5. **Do not cite any of this as a forecast improvement.** It is not one, and today it was not even an evidence change.
