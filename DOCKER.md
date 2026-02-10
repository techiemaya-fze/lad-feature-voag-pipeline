# Vonage Voice Agent - Docker Deployment Guide

## Prerequisites

- Docker Desktop installed and running
- Access to `.env` file with all required credentials
- Access to `secrets/` folder with GCS service account JSON

---

## Quick Start

```bash
# Build and start both services (development)
docker compose up -d --build

# Build and start (production with host networking)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

---

## Required Files

| File/Folder | Purpose | Mount Point in Container |
|-------------|---------|--------------------------|
| `.env` | All environment variables | Loaded via `env_file` |
| `secrets/` | GCS credentials JSON | `/app/secrets/` |

### Directory Structure
```
v2/
├── .env                    # Environment variables (required)
├── secrets/                # GCS credentials (required)
│   └── salesmaya-yts-*.json
├── docker-compose.yml      # Base compose file
├── docker-compose.prod.yml # Production overrides
├── Dockerfile.api          # API image
└── Dockerfile.worker       # Worker image
```

---

## Building Images

### Build Both Images
```bash
docker compose build
```

### Build Individual Images
```bash
# API only
docker build -f Dockerfile.api -t vonage-voice-agent-api:latest .

# Worker only
docker build -f Dockerfile.worker -t vonage-voice-agent-worker:latest .
```

### Build with No Cache (clean rebuild)
```bash
docker compose build --no-cache
```

---

## Running Containers

### Development Mode (Bridge Network)
```bash
docker compose up -d
```

### Production Mode (Host Network for Worker)
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Run Manually (without compose)
```bash
# API
docker run -d \
  --name vonage-api \
  -p 8000:8000 \
  --env-file .env \
  -e GCS_CREDENTIALS_JSON=/app/secrets/salesmaya-yts-6b49f7694826.json \
  -v ./secrets:/app/secrets:ro \
  vonage-voice-agent-api:latest

# Worker (with host network for LiveKit)
docker run -d \
  --name vonage-worker \
  --network host \
  --env-file .env \
  -e GCS_CREDENTIALS_JSON=/app/secrets/salesmaya-yts-6b49f7694826.json \
  -v ./secrets:/app/secrets:ro \
  vonage-voice-agent-worker:latest
```

---

## Managing Containers

### View Running Containers
```bash
docker ps
```

### View Logs
```bash
# Follow logs (live)
docker logs -f vonage-voice-agent-worker
docker logs -f vonage-voice-agent-api

# Last 100 lines
docker logs --tail 100 vonage-voice-agent-worker
```

### Restart Containers
```bash
# Restart without rebuilding
docker compose restart

# Restart with env changes applied
docker compose up -d --force-recreate

# Restart single service
docker compose restart worker
docker compose restart api
```

### Stop Containers
```bash
# Stop all
docker compose down

# Stop with volume cleanup
docker compose down -v
```

---

## Updating After Code Changes

### Quick Update (rebuild + restart)
```bash
docker compose up -d --build
```

### Full Clean Rebuild
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## Updating After .env Changes

Environment variables require container recreation:

```bash
docker compose up -d --force-recreate
```

---

## Image Transfer (Export/Import)

### Save Images to File
```bash
# Save both images to tar files
docker save vonage-voice-agent-api:latest | gzip > vonage-api.tar.gz
docker save vonage-voice-agent-worker:latest | gzip > vonage-worker.tar.gz
```

### Load Images from File
```bash
# On target machine
docker load < vonage-api.tar.gz
docker load < vonage-worker.tar.gz
```

### Push to Registry
```bash
# Tag for registry
docker tag vonage-voice-agent-api:latest your-registry.com/vonage-api:latest
docker tag vonage-voice-agent-worker:latest your-registry.com/vonage-worker:latest

# Push
docker push your-registry.com/vonage-api:latest
docker push your-registry.com/vonage-worker:latest
```

### Pull from Registry
```bash
docker pull your-registry.com/vonage-api:latest
docker pull your-registry.com/vonage-worker:latest
```

---

## Health Checks

```bash
# API health
curl http://localhost:8000/healthz

# Worker health (if not using host network)
curl http://localhost:8081/
```

---

## Troubleshooting

### Container Won't Start
```bash
# Check logs for errors
docker logs vonage-voice-agent-worker

# Check if secrets are mounted
docker exec vonage-voice-agent-worker ls -la /app/secrets/
```

### GCS Credentials Error
Ensure:
1. `secrets/` folder exists with the JSON file
2. Volume mount is correct: `./secrets:/app/secrets:ro`
3. `GCS_CREDENTIALS_JSON` points to `/app/secrets/<filename>.json`

### Recording Issues
- Check LiveKit Cloud dashboard for egress status
- Verify GCS bucket permissions
- Confirm credentials have Storage Object Admin role

### Network Issues (Worker)
For production, always use host networking:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Resource Usage

| Service | CPU | Memory | Notes |
|---------|-----|--------|-------|
| API | 0.5-2 cores | 512MB-2GB | Scales with requests |
| Worker | 1-4 cores | 2-8GB | ~500MB per concurrent call |

---

## Graceful Shutdown

Workers handle long calls. Use proper timeout:
```bash
# Stop with 5-hour timeout (for production)
docker stop -t 18000 vonage-voice-agent-worker
```

The `docker-compose.prod.yml` includes `stop_grace_period: 5h` automatically.
