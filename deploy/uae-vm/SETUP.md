# UAE VM Worker Deployment - Manual Steps

## Files Already Uploaded

All files are at `~/voag-worker/` on uae-vm:

| File | Purpose |
|------|---------|
| `.env` | Environment config (LiveKit, DB, API keys) |
| `docker-compose.worker.yml` | Worker-only compose (3 replicas) |
| `vonage-worker.tar` | Docker image (~4.4GB) |
| `secrets/*.json` | GCS + OAuth credentials |

---

## Steps to Run

### 1. Load the Docker Image

```bash
cd ~/voag-worker
docker load -i vonage-worker.tar
```

This imports `vonage-worker:latest` into Docker. Takes 1-2 minutes.

### 2. Verify Image Loaded

```bash
docker images vonage-worker
```

Should show `vonage-worker   latest   ...   4.4GB` (approx).

### 3. Update MAIN_API_BASE_URL (if needed)

Edit `.env` and set the external API URL that workers call back to:

```bash
nano .env
# Find this line and update if needed:
# MAIN_API_BASE_URL=https://voag.techiemaya.com
```

### 4. Start Workers

```bash
docker compose -f docker-compose.worker.yml up -d
```

### 5. Verify Workers Are Running

```bash
docker ps
```

Expected output — 3 healthy workers:

```
CONTAINER ID  IMAGE               STATUS                    NAMES
abc123        vonage-worker       Up 30s (healthy)          voag-worker-worker-1
def456        vonage-worker       Up 30s (healthy)          voag-worker-worker-2
ghi789        vonage-worker       Up 30s (healthy)          voag-worker-worker-3
```

### 6. Check Logs

```bash
# All workers
docker compose -f docker-compose.worker.yml logs -f

# Single worker
docker compose -f docker-compose.worker.yml logs -f worker-1
```

Look for: `registered worker` — confirms workers connected to LiveKit.

---

## Useful Commands

| Action | Command |
|--------|---------|
| Stop all workers | `docker compose -f docker-compose.worker.yml down` |
| Restart workers | `docker compose -f docker-compose.worker.yml restart` |
| Scale to N workers | Edit `WORKER_REPLICAS=N` in `.env`, then `docker compose -f docker-compose.worker.yml up -d` |
| View logs | `docker compose -f docker-compose.worker.yml logs -f` |
| Check health | `docker ps` |

---

## Architecture Recap

```
┌─────────────────── uae-vm (internal) ──────────────────┐
│                                                         │
│  LiveKit Server (ws://127.0.0.1:7880)                  │
│       │                                                 │
│       ├── Worker 1 (host network)                      │
│       ├── Worker 2 (host network)                      │
│       └── Worker 3 (host network)                      │
│                                                         │
│  SIP Trunk (ST_svHE4RdTc7Ds)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
         ↕ Worker callbacks via MAIN_API_BASE_URL
┌─────────────────── External ────────────────────────────┐
│  API (main.py) @ https://voag.techiemaya.com           │
└─────────────────────────────────────────────────────────┘
```
