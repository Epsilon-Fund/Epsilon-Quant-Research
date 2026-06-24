<<<<<<< HEAD
=======
---
title: "L2 Pipeline — R2 Cloud Backup Setup (Handover)"
created: 2026-06-18
status: active
owner: alvaro
project: mm
para: project
tags: [data, infrastructure, r2, backup, handover]
---

>>>>>>> 7703de6cc61a18ab11cfc528ecd0e18666dedbf5
# L2 Pipeline — R2 Cloud Backup Setup (Handover)

## Context

We have a Polymarket L2 data capture pipeline running 24/7 on a Hetzner VPS. It records live order book data for our market-making research. Everything is working — capture, discovery, compression — but the **cloud backup** (Cloudflare R2) still needs to be set up.

Your task: set up R2 so the data gets backed up to the cloud automatically.

## What's already running

| Component | Status | What it does |
|---|---|---|
| capture.service | Running 24/7 | Records WebSocket data to JSONL.gz files |
| discovery.timer | Every 15 min | Refreshes the list of markets to capture |
| compress.timer | Every hour | Converts JSONL.gz → Parquet |
| sync.timer | **NOT enabled** | Uploads to R2 — needs R2 setup first |

## VPS access

- IP: `89.167.68.98`
- Connect: `ssh root@89.167.68.98`
- You need your SSH public key added first (Carlos will do this — send him the output of `cat ~/.ssh/id_rsa.pub`)
- Pipeline lives at: `/opt/epsilon/l2_ingestion/`

## What you need to do

### 1. Create a Cloudflare account and R2 bucket

1. Go to https://dash.cloudflare.com and sign up (free)
2. Go to **R2** in the sidebar
3. Click **Create bucket**, name it exactly: `epsilon-polymarket-data`
4. Note your **Account ID** (visible in the URL or R2 overview page)

### 2. Create an API token for the bucket

1. In R2, click **Manage R2 API Tokens** → **Create API token**
2. Permissions: **Object Read & Write**, scoped to the `epsilon-polymarket-data` bucket
3. Copy the **Access Key ID** and **Secret Access Key** (shown only once — save them)

### 3. Configure rclone on the VPS

SSH into the server and run:

```bash
rclone config
```

Answer the prompts:
- `n` (new remote) → name: `r2`
- Storage: `s3`
- Provider: `Cloudflare`
- `access_key_id`: paste your Access Key ID
- `secret_access_key`: paste your Secret Access Key
- `region`: `auto`
- `endpoint`: `https://<YOUR_ACCOUNT_ID>.r2.cloudflarestorage.com`
- Accept defaults for everything else → `y` to confirm → `q` to quit

### 4. Test the connection

```bash
rclone lsd r2:epsilon-polymarket-data
```

Should return empty (no error). If it errors, the credentials or endpoint are wrong.

### 5. Do a manual sync test

```bash
cd /opt/epsilon/l2_ingestion
bash sync/sync_cloud.sh
```

Check that data appeared in R2:

```bash
rclone lsd r2:epsilon-polymarket-data
```

You should see `parquet/` and `raw/` folders.

### 6. Enable the automatic sync timer

```bash
systemctl enable --now sync.timer
```

This will run the sync every 6 hours automatically.

### 7. Verify everything is healthy

```bash
cd /opt/epsilon/l2_ingestion
./venv/bin/python monitoring/health_check.py
```

All components should be GREEN (including sync now).

## If something goes wrong

Check logs:
```bash
journalctl -u sync.service --since "1 hour ago"
journalctl -u capture.service -f  # live capture log, Ctrl+C to stop watching
```

## What NOT to touch

- Do not modify or restart `capture.service` — it's recording live data
- Do not delete anything in `data/raw/` or `data/parquet/` manually
- Do not touch `/opt/epsilon/` files outside of `l2_ingestion/`
- Do not touch `epsilon-dashboard.service` — it's a separate project
