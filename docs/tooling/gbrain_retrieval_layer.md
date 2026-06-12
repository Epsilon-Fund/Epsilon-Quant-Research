# gbrain — local read-only retrieval layer over the vault

A **disposable, local** semantic + graph index on top of our existing Markdown vault (github.com/garrytan/gbrain). Our `.md` + hubs + basename-wikilink graph remain the sole source of record; gbrain is read-only on the files and lives entirely in `~/.gbrain`. Set up 2026-06-10.

## What it gives us / what it does NOT
- **Local semantic search** (`gbrain search`) — embeds the query with a local model, returns ranked source chunks + citations. Finds notes by *meaning* (grep needs exact terms).
- **Graph traversal** (`gbrain graph <slug>`, `gbrain backlinks <slug>`) — our `[[basename]]` links became 1284 edges via `link_resolution.global_basename`.
- **NO local synthesis.** gbrain's `think` synthesis is **cloud-only by design** (every local provider — ollama/llama-server/litellm — is embeddings-only; chat is gated to anthropic/openai/etc.). To honor the privacy rule we did **not** wire a cloud LLM, so `gbrain think` degrades to *gather-only* and never leaves the machine. In MCP use, let Claude Code/Codex synthesize over gbrain's local retrieval.

## Local embedder
`ollama:nomic-embed-text` (768-dim, no API key). Requires the Ollama server running — launch **Ollama.app** (it runs `localhost:11434` persistently).

## Intentionally disabled (read-only posture)
Dream cycle, cron/autopilot enrichment, `sync --watch`, skillpacks, schema-pack mutation, and any write-back — none installed/enabled. `models.think`/`models.default` are pinned to `ollama:*` as a **fail-safe** so an accidental `think` can't fall back to cloud. The pglite DB exposes write tools over MCP but they touch only the disposable DB, never the `.md`.

## Exact commands run
```bash
curl -fsSL https://bun.sh/install | bash                       # bun runtime
# Ollama.app -> /Applications; `ollama serve`; ollama pull nomic-embed-text
bun install -g github:garrytan/gbrain
cd ~ && gbrain init --pglite --embedding-model ollama:nomic-embed-text   # state in ~/.gbrain
gbrain import /Users/justiniturregui/Desktop/github/epsilon-quant-research
gbrain config set link_resolution.global_basename true
gbrain extract links            # build wikilink graph edges (NOTE: `--stale` extracts 0 on first run)
gbrain doctor                   # confirms embedder=local, links current
# MCP wiring (local stdio):
codex mcp add gbrain --env PATH="$HOME/.bun/bin:/usr/bin:/bin" -- "$HOME/.bun/bin/gbrain" serve
# Claude Code (no standalone CLI here): gbrain stdio server added to ~/.claude.json
#   projects[<repo>].mcpServers.gbrain  (command=~/.bun/bin/gbrain, args=[serve], env.PATH includes ~/.bun/bin)
```

## Re-import after vault changes
```bash
gbrain import /Users/justiniturregui/Desktop/github/epsilon-quant-research && gbrain extract links
```

## Full teardown (leaves zero trace in the vault)
```bash
codex mcp remove gbrain
# Claude: delete the "gbrain" key under projects[<repo>].mcpServers in ~/.claude.json
#   (or restore ~/.claude.json.pre-gbrain.bak)
bun remove -g gbrain
rm -rf ~/.gbrain
# optional: rm ~/.local/bin/ollama, remove /Applications/Ollama.app + `ollama rm nomic-embed-text`, uninstall bun
```
