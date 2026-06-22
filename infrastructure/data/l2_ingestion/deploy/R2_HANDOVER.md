# L2 Pipeline — R2 Cloud Backup Setup

## Context

We have a Polymarket L2 data capture pipeline running 24/7 on this VPS. It records live order book data for our market-making research. Everything is working — capture, discovery, compression — but the **cloud backup** (Cloudflare R2) still needs to be set up.

Your task: set up R2 so the data gets backed up to the cloud automatically.

## What's already running

| Component | Status | What it does |
|---|---|---|
| capture.service | Running 24/7 | Records WebSocket data to JSONL.gz files |
| discovery.timer | Every 15 min | Refreshes the list of markets to capture |
| compress.timer | Every hour | Converts JSONL.gz to Parquet |
| sync.timer | **NOT enabled yet** | Will upload to R2 — needs R2 setup first |

Pipeline lives at: `/opt/epsilon/l2_ingestion/`

## Step 1 — Create a Cloudflare account and R2 bucket

1. Go to https://dash.cloudflare.com and sign up (free tier is enough)
2. Go to **R2** in the sidebar
3. Click **Create bucket**, name it exactly: **epsilon-polymarket-data**
4. Note your **Account ID** (visible in the URL or R2 overview page). The S3 endpoint will be: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

## Step 2 — Create an API token for the bucket

1. In R2, click **Manage R2 API Tokens** → **Create API token**
2. Permissions: **Object Read & Write**, scoped to the `epsilon-polymarket-data` bucket
3. Copy the **Access Key ID** and **Secret Access Key** (shown only once — save them somewhere safe)

## Step 3 — Configure rclone on the VPS

SSH in and run:

```bash
ssh root@89.167.68.98
rclone config
```

Answer the prompts:
- `n` (new remote) → name: **r2**
- Storage: **s3**
- Provider: **Cloudflare**
- `access_key_id`: paste your Access Key ID
- `secret_access_key`: paste your Secret Access Key
- `region`: **auto**
- `endpoint`: `https://<YOUR_ACCOUNT_ID>.r2.cloudflarestorage.com`
- Accept defaults for everything else → `y` to confirm → `q` to quit

## Step 4 — Test the connection

```bash
rclone lsd r2:epsilon-polymarket-data
```

Should return empty with no errors. If it errors, double-check the credentials and endpoint.

## Step 5 — Do a manual sync test

```bash
cd /opt/epsilon/l2_ingestion
bash sync/sync_cloud.sh
```

Then check that data appeared in R2:

```bash
rclone ls r2:epsilon-polymarket-data/parquet/ | head -10
```

You should see `.parquet` files listed.

## Step 6 — Enable the automatic sync timer

```bash
systemctl enable --now sync.timer
systemctl list-timers | grep sync
```

This runs the sync every 6 hours automatically. You should see the next scheduled run time.

## Step 7 — Verify everything is healthy

```bash
cd /opt/epsilon/l2_ingestion
./venv/bin/python monitoring/health_check.py
```

All 5 components should be GREEN now (including sync).

## Troubleshooting

Check sync logs:
```bash
journalctl -u sync.service --since "1 hour ago"
```

Check capture logs (live):
```bash
journalctl -u capture.service -f
```
(Ctrl+C to stop watching — this does NOT stop the capture daemon)

Check all timers at a glance:
```bash
systemctl list-timers --all | grep -E 'discovery|compress|sync'
```

## What NOT to touch

- **Do not restart capture.service** unless there's a RED alert — it's recording live data and a restart means a brief gap
- **Do not delete** anything in `data/raw/` or `data/parquet/` manually — the sync script handles cleanup
- **Do not touch** `/opt/epsilon/` files outside of `l2_ingestion/` — there's a separate dashboard running
- **Do not touch** `epsilon-dashboard.service`
