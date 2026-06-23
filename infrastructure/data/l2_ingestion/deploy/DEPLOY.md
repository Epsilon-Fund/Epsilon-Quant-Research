# Deploying the L2 Ingestion Pipeline on the Hetzner VPS

Step-by-step guide to run the Polymarket L2 capture pipeline 24/7 on our existing
server. You don't need to be a sysadmin — just follow the commands in order.

## Plain-English overview

We're putting four small programs on a rented Linux server and having the server
run them automatically forever:

- **capture** — runs constantly, recording live order-book data to disk.
- **discovery** — runs every 15 min, refreshing the list of markets to record.
- **compression** — runs hourly, converting raw files to compact Parquet.
- **sync** — runs every 6 h, uploading to Cloudflare R2 (cloud backup) and pruning old local raw.

A piece of Linux called **systemd** keeps them running and restarts capture if it
crashes. We talk to the server over SSH.

## The server

| | |
|---|---|
| Provider | Hetzner CX23 — 2 vCPU, 4 GB RAM, 40 GB disk |
| OS | Ubuntu 24.04 |
| IP | `89.167.68.98` |
| Hostname | `Midas` (Helsinki) |
| Access | SSH as `root` (already working) |

> **DO NOT TOUCH** the existing `/opt/epsilon/` repo clone or the running
> `epsilon-dashboard.service`. Everything below lives in a **new** subfolder,
> `/opt/epsilon/l2_ingestion/`, and uses its own systemd units. We never modify
> or restart the dashboard.

Throughout: commands prefixed `local$` run on **your laptop**; commands prefixed
`vps#` run on the **server** (after `ssh root@89.167.68.98`).

---

## Step 1 — Check Python (need 3.11+)

```bash
vps# python3 --version
```

Ubuntu 24.04 ships Python 3.12, so this should already be fine. If for some
reason it's < 3.11:

```bash
vps# apt update && apt install -y python3
```

## Step 2 — Install system packages (pip, venv, rclone)

```bash
vps# apt update
vps# apt install -y python3-pip python3-venv rclone
vps# rclone version    # confirm rclone installed
```

(`python3-venv` is what lets us create the isolated environment in Step 5.)

## Step 3 — Create the deploy directory

```bash
vps# mkdir -p /opt/epsilon/l2_ingestion
```

This is a sibling of the existing dashboard files under `/opt/epsilon/`, not a
replacement. We only ever write inside `l2_ingestion/`.

## Step 4 — Copy the pipeline code from the repo

Run these from the repo root **on your laptop**. The first rsync copies the
pipeline; the second vendors the `data_infra` package that the discovery daemon
imports (the `GammaClient`). We exclude local test data, the venv, and caches.

```bash
local$ rsync -av --exclude 'data/' --exclude 'venv/' --exclude '__pycache__/' \
  infrastructure/data/l2_ingestion/ \
  root@89.167.68.98:/opt/epsilon/l2_ingestion/

local$ rsync -av --exclude '__pycache__/' \
  polymarket/research/data_infra/ \
  root@89.167.68.98:/opt/epsilon/l2_ingestion/vendor/data_infra/
```

After this the server should have:

```
/opt/epsilon/l2_ingestion/
├── config/universes.yaml
├── discovery/  capture/  compression/  sync/  monitoring/  deploy/
├── requirements.txt
└── vendor/data_infra/gamma.py     ← vendored GammaClient
```

> **Why vendor/?** discovery does `from data_infra.gamma import GammaClient`. On
> the VPS the repo isn't present, so we copy just that module and point
> `PYTHONPATH` at `vendor/` (already set in `discovery.service`).

## Step 5 — Create the venv and install dependencies

```bash
vps# cd /opt/epsilon/l2_ingestion
vps# python3 -m venv venv
vps# ./venv/bin/pip install --upgrade pip
vps# ./venv/bin/pip install -r requirements.txt
```

This creates an isolated Python at `/opt/epsilon/l2_ingestion/venv/` (the path
the systemd units expect) and installs websocket-client, pyyaml, pandas, pyarrow,
httpx, tenacity into it — without touching the system Python or the dashboard.

## Step 6 — Cloudflare R2 setup

**6a. Create the bucket (in a browser):**
1. Sign in at <https://dash.cloudflare.com> → **R2**.
2. **Create bucket** → name it exactly **`epsilon-polymarket-data`** → create.
3. Note your **Account ID** (shown in the R2 overview / URL). The S3 endpoint is
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

**6b. Create an API token:**
1. R2 → **Manage R2 API Tokens** → **Create API token**.
2. Permissions: **Object Read & Write**, scoped to the `epsilon-polymarket-data`
   bucket.
3. Copy the **Access Key ID** and **Secret Access Key** (shown once).

**6c. Configure the rclone remote named `r2` on the server:**

```bash
vps# rclone config
```
Answer the prompts:
- `n` (new remote) → name: **`r2`**
- Storage: **`s3`**
- Provider: **`Cloudflare`**
- `access_key_id`: *(paste Access Key ID)*
- `secret_access_key`: *(paste Secret Access Key)*
- `region`: `auto`
- `endpoint`: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
- accept defaults for the rest → `y` to confirm → `q` to quit.

> Secrets live only in `/root/.config/rclone/rclone.conf` on the server — never in
> the repo. The resulting config looks like (secrets redacted):
> ```ini
> [r2]
> type = s3
> provider = Cloudflare
> access_key_id = ********
> secret_access_key = ********
> endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
> ```

**6d. Test it:**
```bash
vps# rclone lsd r2:epsilon-polymarket-data    # should list (empty) without error
```

## Step 7 — First-run verification (before installing services)

Prove each stage works manually. Run from the deploy dir:

```bash
vps# cd /opt/epsilon/l2_ingestion

# 1) discovery — writes data/live_universe.json (needs PYTHONPATH for GammaClient)
vps# PYTHONPATH=vendor ./venv/bin/python discovery/daemon.py --once

# 2) capture — record for 60s, then it stops itself
vps# ./venv/bin/python capture/daemon.py --duration-seconds 60

# 3) compression — convert today's shards (force = skip the 1h age check)
vps# ./venv/bin/python compression/pipeline.py --input data/raw/$(date -u +%F)/ --force

# 4) sync — upload to R2 and report
vps# bash sync/sync_cloud.sh

# 5) health check — should be mostly GREEN right after the above
vps# ./venv/bin/python monitoring/health_check.py
```

If discovery wrote a `live_universe.json`, capture produced `.jsonl.gz` files
under `data/raw/<today>/`, compression made `.parquet` files under
`data/parquet/<today>/`, and `rclone lsd r2:epsilon-polymarket-data` now shows
`parquet/` and `raw/` — you're ready to install the services.

## Step 8 — Install the systemd unit files

```bash
vps# cp /opt/epsilon/l2_ingestion/deploy/*.service /etc/systemd/system/
vps# cp /opt/epsilon/l2_ingestion/deploy/*.timer   /etc/systemd/system/
vps# systemctl daemon-reload
```

(`daemon-reload` tells systemd to read the new unit files.)

## Step 9 — Enable timers and start capture for production

```bash
# the always-on daemon: start now AND on every boot
vps# systemctl enable --now capture.service

# the periodic jobs: enable the TIMERS (not the .service units)
vps# systemctl enable --now discovery.timer
vps# systemctl enable --now compress.timer
vps# systemctl enable --now sync.timer
```

Confirm:
```bash
vps# systemctl status capture.service        # should say "active (running)"
vps# systemctl list-timers --all             # shows next run time for each timer
```

> To kick a timer-driven job immediately (instead of waiting):
> `systemctl start discovery.service`

## Step 10 — Checking logs (journalctl)

All output goes to the systemd journal:

```bash
# follow capture live (Ctrl-C to stop watching — does NOT stop the daemon)
vps# journalctl -u capture.service -f

# last hour of a given service
vps# journalctl -u discovery.service --since "1 hour ago"
vps# journalctl -u compress.service --since "1 hour ago"
vps# journalctl -u sync.service --since "today"

# only errors
vps# journalctl -u capture.service -p err

# disk used by the journal (it self-rotates, but good to know)
vps# journalctl --disk-usage
```

## Step 11 — Running the health check

```bash
vps# cd /opt/epsilon/l2_ingestion
vps# ./venv/bin/python monitoring/health_check.py
```

Reads one line per component:

- **GREEN** — working as designed, nothing to do.
- **YELLOW** — degraded but not broken; keep an eye on it (may self-recover).
- **RED** — broken, act now (data is being lost or not backed up).

What each line means:
- **capture** RED → no fresh shards in >2h; check `systemctl status capture` and
  `journalctl -u capture -p err`.
- **gaps** RED → many WS disconnects/stale warnings in 24h; network or WS issues.
- **compression** RED → completed shards piling up unprocessed; check
  `journalctl -u compress`.
- **sync** RED → cloud sync hasn't run in hours; check `journalctl -u sync` and
  `rclone lsd r2:epsilon-polymarket-data`.
- **inventory** → informational (days captured, parquet/raw size).

Machine-readable form for alerting (exit code 0/1/2 = GREEN/YELLOW/RED):
```bash
vps# ./venv/bin/python monitoring/health_check.py --json
vps# ./venv/bin/python monitoring/health_check.py --exit-code; echo "exit=$?"
```

---

## Quick reference — common operations

```bash
# stop / start / restart capture (graceful: closes files cleanly)
vps# systemctl stop capture.service
vps# systemctl start capture.service
vps# systemctl restart capture.service

# see all our units at a glance
vps# systemctl list-timers --all | grep -E 'discovery|compress|sync'
vps# systemctl status capture.service

# after editing a unit file (re-copy it, then):
vps# systemctl daemon-reload && systemctl restart <unit>

# update the code later (from laptop), then restart capture:
local$ rsync -av --exclude 'data/' --exclude 'venv/' --exclude '__pycache__/' \
  infrastructure/data/l2_ingestion/ root@89.167.68.98:/opt/epsilon/l2_ingestion/
vps# systemctl restart capture.service
```

## Disk safety

On the 40 GB disk, raw is the buffer: ~1.2 GB/day raw, pruned locally after 7
days by `sync_cloud.sh` (cloud keeps the backup via `rclone copy`). Parquet stays
local for querying. If disk ever fills, check `du -sh /opt/epsilon/l2_ingestion/data/*`
and confirm sync is running (Step 11).
