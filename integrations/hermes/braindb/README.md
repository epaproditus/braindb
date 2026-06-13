# BrainDB memory provider for Hermes Agent

Use a self-hosted [BrainDB](https://github.com/dimknaf/braindb) as the long-term
memory backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It's a small native Hermes `MemoryProvider`: the Hermes agent gets two tools that
talk to a running BrainDB over HTTP. Nothing about BrainDB's API is re-implemented
here — the gateway tool delegates to BrainDB's own agent, which already wields its
full toolset.

## Install

**Option A — drop-in folder** (user plugins live FLAT under `plugins/<name>/`)

```bash
cp -r braindb ~/.hermes/plugins/braindb
```

**Option B — pip** (if packaged): `pip install hermes-plugins-braindb`.

Then activate it:

```bash
hermes memory setup        # pick "braindb"
```

## Configure

One setting, resolved in this order: `BRAINDB_BASE_URL` env var →
`$HERMES_HOME/braindb.json` (`{"base_url": "..."}`) → default `http://localhost:8000`.

```bash
export BRAINDB_BASE_URL=http://localhost:8000
```

## Tools the agent gets

- **`braindb_ask(query)`** — ask BrainDB in natural language to recall or remember.
  This is the gateway to *all* of BrainDB: its internal agent does the search,
  storage, graph-linking and reasoning, and returns an answer.
- **`braindb_ingest(file_path)`** — upload a local text file to BrainDB. BrainDB
  ingests and **fact-extracts it asynchronously**, so the extracted facts show up
  on a *later* `braindb_ask` recall, not immediately. Supported extensions:
  `.md .txt .json .yaml .yml .csv .log .html .xml`.

Usage instructions for the agent are loaded **live** from BrainDB's
`/api/v1/skill/braindb-agent` endpoint (single source of truth, cached locally),
so there's no duplicated prompt to keep in sync.

## Deployment notes

- **Shared store, single user.** All memory is one BrainDB instance; this provider
  does no per-user scoping.
- **BrainDB has no authentication.** Run Hermes and BrainDB **co-located**, with
  BrainDB on loopback (`localhost`). For a *remote* BrainDB you must put your own
  auth / tunnel in front of it — that's out of scope here.
- The provider makes all calls over HTTP itself; it never shells out via the
  agent's terminal, so it works regardless of Hermes' terminal sandbox.

## Requires on the BrainDB side

Two small additive endpoints (shipped with BrainDB in
`braindb/routers/integrations.py`): `GET /api/v1/skill/{name}` (serves the skill
text) and `POST /api/v1/entities/datasources/upload` (lands a file in
`data/sources/` for the watcher to ingest). The BrainDB **watcher** must be
running for ingestion to happen (it is by default in `docker-compose`).
