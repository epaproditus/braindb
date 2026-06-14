# Hermes ↔ BrainDB sandbox (run it on an isolated throwaway host)

A self-contained way to actually run **Hermes Agent** against BrainDB and test the
[`braindb` memory provider](../braindb/) — **safely**.

## ⚠ Why a throwaway host

Hermes is an autonomous agent: by default its `terminal` / `code_execution` / browser tools
run **on the host**, and its docs say plainly *"the only security boundary against an
adversarial LLM is the OS."* So, defense in depth:

1. We **strip** the dangerous toolsets in `config.yaml` — the agent can only use `braindb_*`.
2. We run Hermes **in a container**, as a non-root UID. (We don't `cap_drop` everything — the
   image's s6 init needs `CHOWN`/`SETUID`/`SETGID`; that's why the host isolation matters most.)
3. Most importantly, we run the whole thing on a **throwaway, low-trust host** (a Raspberry
   Pi) — never a work laptop. If Hermes misbehaves, the blast radius is the Pi.

We also run a **separate, throwaway BrainDB** here — not your real personal one — so the test
never pollutes real memory or re-exposes an unauthenticated service.

## What runs

- **BrainDB** (api + bundled Postgres + watcher) via BrainDB's own compose, `internal-db`
  profile, `LLM_PROFILE=deepinfra`, on the `local-network` docker network.
- **Hermes** (`nousresearch/hermes-agent`, this folder's compose) on the same network,
  reaching BrainDB at `http://braindb_api:8000`, brain on **deepinfra**.

Both LLMs are on deepinfra — **the workstation is never touched.**

## Deploy (on the Pi)

```bash
cd ~/braindb
docker network create local-network 2>/dev/null || true

# 1) BrainDB env (repo-root .env): bundled DB + deepinfra + wiki off
cat > .env <<'EOF'
COMPOSE_PROFILES=internal-db
LLM_PROFILE=deepinfra
DEEPINFRA_API_KEY=<your-deepinfra-key>
WIKI_ENABLED=false
EOF

# 2) Build + start BrainDB (first arm64 build is slow; downloads the embedding model)
docker compose up -d --build api watcher braindb_db
curl -s http://localhost:8000/health        # {"status":"ok","embeddings":true}

# 3) Hermes profile: copy the template, then set YOUR model + key in it (gitignored)
cd integrations/hermes/sandbox
mkdir -p hermes-data
cp config.yaml hermes-data/config.yaml
#   edit hermes-data/config.yaml ->  model.default: <a >=64k-context model id>
#                                     model.api_key: <your deepinfra key>

# 4) Start Hermes
docker compose -f docker-compose.hermes-test.yml up -d
```

## Verify

```bash
docker exec hermes_test curl -s http://braindb_api:8000/health   # Hermes can reach BrainDB
docker exec hermes_test hermes memory status                     # Provider: braindb / installed + active

# Talk to it (-q = single non-interactive query). It should call braindb_ask.
docker exec hermes_test hermes chat -q "Remember that my favourite colour is teal, then tell me what you stored."

# Ingest a local file -> braindb_ingest -> BrainDB watcher extracts (async)
docker exec hermes_test sh -c 'echo "Project Falcon ships on Friday." > /tmp/note.md'
docker exec hermes_test hermes chat -q "Ingest the file at /tmp/note.md into memory."
```

## Teardown

```bash
docker compose -f docker-compose.hermes-test.yml down -v && rm -rf hermes-data
docker compose down -v        # at the repo root, to drop the throwaway BrainDB
```

## Notes

- **Model:** Hermes refuses models under 64k context; set a ≥64k model in `config.yaml`.
- **Plugin path is FLAT:** the provider is mounted at `/opt/data/plugins/braindb` (= user
  plugins live at `~/.hermes/plugins/<name>/`, not under a `memory/` subdir).
- Remote BrainDB would need auth in front of it; here everything is co-located on one
  isolated host, so loopback / internal-network access is fine.
