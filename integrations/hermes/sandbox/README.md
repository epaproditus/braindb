# Hermes ↔ BrainDB sandbox (run it on an isolated throwaway host)

A self-contained way to actually run **Hermes Agent** against BrainDB to test the
[`braindb` memory provider](../braindb/) — **safely**.

## ⚠ Why a throwaway host

Hermes is an autonomous agent: by default its `terminal` / `execute_code` / browser tools
run **on the host**, and its docs say plainly *"the only security boundary against an
adversarial LLM is the OS."* So:

1. We run Hermes **in a container** with `cap_drop: ALL` + `no-new-privileges`.
2. We **strip** the dangerous toolsets in `config.yaml` (it can only use `braindb_*`).
3. Most importantly, we run the whole thing on a **throwaway, low-trust host** (a
   Raspberry Pi) — never a work laptop. If Hermes misbehaves, the blast radius is the Pi.

We also run a **separate, throwaway BrainDB** here — not your real personal BrainDB — so the
test never pollutes real memory or re-exposes an unauthenticated service.

## What runs

- **BrainDB** (api + bundled Postgres + watcher) via BrainDB's own compose, `internal-db`
  profile, `LLM_PROFILE=deepinfra`. On the `local-network` docker network.
- **Hermes** (`nousresearch/hermes-agent`, this folder's compose) joined to `local-network`,
  reaching BrainDB at `http://braindb_api:8000`, brain on **deepinfra** (a ≥64k model).

Both LLMs are on deepinfra — **the workstation/bench is never touched.**

## Deploy (on the Pi)

```bash
# 0) Get the repo onto the Pi (e.g. rsync from your machine) and cd into it.
cd ~/braindb

# 1) Network BrainDB expects
docker network create local-network 2>/dev/null || true

# 2) BrainDB env (repo root .env) — bundled DB + deepinfra + wiki off
cat > .env <<'EOF'
COMPOSE_PROFILES=internal-db
LLM_PROFILE=deepinfra
DEEPINFRA_API_KEY=<your-deepinfra-key>
WIKI_ENABLED=false
EOF

# 3) Build + start BrainDB (first build on arm64 takes a while; downloads the
#    embedding model on first /health)
docker compose up -d --build
curl -s http://localhost:8000/health    # {"status":"ok","embeddings":true}

# 4) Hermes profile dir: config + .env + (plugin is bind-mounted by the compose)
cd integrations/hermes/sandbox
mkdir -p hermes-data
cp config.yaml hermes-data/config.yaml
cp .env.example hermes-data/.env
#   edit hermes-data/.env -> set CUSTOM_API_KEY=<your-deepinfra-key>

# 5) Start Hermes
docker compose -f docker-compose.hermes-test.yml up -d
```

## Verify

```bash
# Hermes can reach BrainDB
docker exec hermes_test curl -s http://braindb_api:8000/health

# Only the braindb tools are present; shell/code/browser are OFF
docker exec -it hermes_test hermes tools

# braindb is the active memory provider
docker exec hermes_test hermes memory status

# Talk to it — it should call braindb_ask -> BrainDB answers
docker exec -it hermes_test hermes chat -m "What do you remember about me? Save: my favourite colour is teal."

# Ingest: a file INSIDE the container -> braindb_ingest -> BrainDB watcher extracts (async)
docker exec hermes_test sh -c 'echo "Project Falcon ships on Friday." > /tmp/note.md'
docker exec -it hermes_test hermes chat -m "Ingest the file at /tmp/note.md into memory."
```

## Teardown

```bash
docker compose -f docker-compose.hermes-test.yml down -v
rm -rf hermes-data
# (and `docker compose down -v` at the repo root to drop the throwaway BrainDB)
```

## Notes

- Hermes' brain needs a **≥64k-context** model (it refuses smaller). The default
  `meta-llama/Llama-3.3-70B-Instruct` (128k) qualifies; change it in `config.yaml`.
- If `hermes memory status` doesn't show `braindb`, the plugin may need enabling
  (`docker exec -it hermes_test hermes plugins`); the provider is at
  `/opt/data/plugins/memory/braindb` (bind-mounted read-only).
- Remote BrainDB would need auth in front of it; here everything is co-located on one
  isolated host, so loopback/internal-network access is fine.
