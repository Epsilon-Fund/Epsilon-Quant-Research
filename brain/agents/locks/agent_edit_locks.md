---
title: Agent Edit Locks
created: 2026-06-08
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
tags:
  - agent-lane
  - collaboration
  - locks
---

# Agent Edit Locks

Hub links: [[VAULT_MAP]] | [[codex_lane]] | [[cowork_lane]]

This folder is for temporary cooperative edit locks created by `tools/brain_edit_guard.py`.

The volatile `*.lock.md` files are ignored by Git but sync through Relay because `brain/` is shared. That lets another agent see that a canonical Markdown file is being edited. Do not hand-edit lock files. Acquire and release them with the tool.
