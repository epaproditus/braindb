"""
BrainDB memory provider for Hermes Agent (https://github.com/NousResearch/hermes-agent).

Install: drop this `braindb/` folder into `~/.hermes/plugins/braindb/` (user
plugins live FLAT under plugins/<name>/), then `hermes memory setup` -> "braindb".

Design (deliberately small):
- AGENT-ONLY gateway. `braindb_ask` forwards a natural-language request to
  BrainDB's own `/agent/query`, whose internal agent wields BrainDB's full
  toolset — so the *whole* of BrainDB (recall / save / relate / reason) is
  reachable through one tool, with zero duplication of its logic here.
- `braindb_ingest` uploads a local file to BrainDB; BrainDB's watcher then
  ingests + fact-extracts it asynchronously.
- Usage instructions are loaded LIVE from BrainDB's `/skill` endpoint (single
  source of truth, cached locally) — no copied/duplicated prompt — plus a small
  Hermes-specific note.

The provider makes all HTTP calls itself (via `requests`); it never relies on
the agent's shell/terminal. Co-locate Hermes and BrainDB (BrainDB has no auth —
keep it on loopback or behind your own auth/tunnel for remote use).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000"
_SKILL_NAME = "braindb-agent"
_ASK_TIMEOUT = int(os.environ.get("BRAINDB_ASK_TIMEOUT", "600"))  # /agent/query LLM loop; override via env
_HTTP_TIMEOUT = 15     # skill fetch / file upload
_HEALTH_TIMEOUT = 2    # readiness ping

# Last-resort instructions, used ONLY if BrainDB is unreachable AND no cached
# skill exists. The live skill (loaded in initialize) is the real source.
_BUILTIN_INSTRUCTIONS = (
    "BrainDB is your persistent memory. Use `braindb_ask` with a natural-language "
    "request to recall facts/context or to remember something. Use `braindb_ingest` "
    "with a local file path to add a document to memory."
)

# Hermes-specific addendum — net-new glue, NOT a copy of the skill.
_HERMES_NOTES = (
    "\n\n## Using BrainDB from Hermes\n"
    "- `braindb_ask(query)` is the main interface: ask in plain language to recall or "
    "save; BrainDB's own agent does the work and returns an answer.\n"
    "- `braindb_ingest(file_path)` uploads a local file to BrainDB; ingestion + fact "
    "extraction run asynchronously, so new facts appear on a *later* `braindb_ask` "
    "recall, not immediately.\n"
)

_ASK_SCHEMA = {
    "name": "braindb_ask",
    "description": (
        "Query your BrainDB persistent memory in natural language. Use it to RECALL "
        "(facts, context, past decisions — anything you may have stored) and to REMEMBER "
        "(save a new fact/thought, connect ideas). BrainDB's own agent handles search, "
        "storage, graph links and reasoning, and returns a written answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural-language question or instruction, e.g. 'what do we know "
                    "about project X?' or 'remember that the deadline is Friday'."
                ),
            },
        },
        "required": ["query"],
    },
}

_INGEST_SCHEMA = {
    "name": "braindb_ingest",
    "description": (
        "Add a local document to BrainDB memory. Provide the path to a text file on this "
        "machine; it is uploaded to BrainDB, which ingests it and extracts facts in the "
        "background — so the extracted facts become available on a LATER braindb_ask "
        "recall, not instantly. Supported: .md .txt .json .yaml .yml .csv .log .html .xml."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to a local file to ingest.",
            },
        },
        "required": ["file_path"],
    },
}


class BrainDBMemoryProvider(MemoryProvider):
    """Exposes a self-hosted BrainDB instance to Hermes as a memory provider."""

    def __init__(self) -> None:
        self._base_url = _DEFAULT_BASE_URL
        self._instructions = _BUILTIN_INSTRUCTIONS
        self._hermes_home = ""

    @property
    def name(self) -> str:
        return "braindb"

    def is_available(self) -> bool:
        """True only if a BrainDB is actually answering — so Hermes won't
        activate a dead provider. Bounded by a short timeout."""
        try:
            r = requests.get(f"{self._resolve_base_url()}/health", timeout=_HEALTH_TIMEOUT)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = kwargs.get("hermes_home", "") or ""
        self._base_url = self._resolve_base_url()
        self._instructions = self._load_instructions()

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "BrainDB base URL (the running BrainDB API).",
                "default": _DEFAULT_BASE_URL,
                "env_var": "BRAINDB_BASE_URL",
            },
        ]

    def save_config(self, values, hermes_home: str) -> None:
        try:
            data = {k: v for k, v in (values or {}).items() if v is not None}
            (Path(hermes_home) / "braindb.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:  # config persistence is best-effort
            logger.warning("braindb: could not write config: %s", e)

    def system_prompt_block(self) -> str:
        return f"{self._instructions}{_HERMES_NOTES}"

    def get_tool_schemas(self):
        return [_ASK_SCHEMA, _INGEST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            if tool_name == "braindb_ask":
                return self._ask(args)
            if tool_name == "braindb_ingest":
                return self._ingest(args)
            return _err(f"Unknown tool: {tool_name}")
        except Exception as e:  # a tool call must never crash the turn
            logger.exception("braindb: tool %s failed", tool_name)
            return _err(str(e))

    # ---- internals -------------------------------------------------------

    def _resolve_base_url(self) -> str:
        """env BRAINDB_BASE_URL → $HERMES_HOME/braindb.json → default."""
        env = os.environ.get("BRAINDB_BASE_URL")
        if env:
            return env.rstrip("/")
        home = self._hermes_home or os.environ.get("HERMES_HOME", "")
        if home:
            cfg = Path(home) / "braindb.json"
            if cfg.is_file():
                try:
                    val = json.loads(cfg.read_text(encoding="utf-8")).get("base_url")
                    if val:
                        return str(val).rstrip("/")
                except Exception:
                    pass
        return _DEFAULT_BASE_URL

    def _cache_path(self) -> Path:
        home = self._hermes_home or os.environ.get(
            "HERMES_HOME", str(Path.home() / ".hermes")
        )
        return Path(home) / "braindb_skill.md"

    def _load_instructions(self) -> str:
        """Single source of truth: fetch the skill from BrainDB, cache it so a
        transient outage still has the last copy, fall back to a built-in line.
        Only needed when BrainDB is up (the tools call BrainDB anyway), so this
        introduces no new failure mode."""
        try:
            r = requests.get(
                f"{self._base_url}/api/v1/skill/{_SKILL_NAME}", timeout=_HTTP_TIMEOUT
            )
            if r.status_code == 200 and r.text.strip():
                try:
                    self._cache_path().write_text(r.text, encoding="utf-8")
                except Exception:
                    pass
                return r.text
        except requests.RequestException:
            pass
        try:
            cached = self._cache_path()
            if cached.is_file():
                return cached.read_text(encoding="utf-8")
        except Exception:
            pass
        return _BUILTIN_INSTRUCTIONS

    def _ask(self, args: dict) -> str:
        query = (args or {}).get("query", "").strip()
        if not query:
            return _err("Missing required parameter: query")
        try:
            r = requests.post(
                f"{self._base_url}/api/v1/agent/query",
                json={"query": query},
                timeout=_ASK_TIMEOUT,
            )
            r.raise_for_status()
        except requests.Timeout:
            return _err(f"BrainDB agent timed out after {_ASK_TIMEOUT}s; try a simpler query.")
        except requests.RequestException as e:
            return _err(f"BrainDB request failed: {e}")
        answer = (r.json() or {}).get("answer", "")
        return json.dumps({"answer": answer})

    def _ingest(self, args: dict) -> str:
        raw = (args or {}).get("file_path", "").strip()
        if not raw:
            return _err("Missing required parameter: file_path")
        path = Path(raw).expanduser()
        if not path.is_file():
            return _err(f"File not found: {raw}")
        try:
            data = path.read_bytes()
            r = requests.post(
                f"{self._base_url}/api/v1/entities/datasources/upload",
                params={"filename": path.name},
                data=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            return _err(f"BrainDB upload failed: {e}")
        body = r.json() if r.content else {}
        return json.dumps({
            "status": "uploaded",
            "filename": body.get("filename", path.name),
            "message": "Uploaded; BrainDB will ingest and fact-extract it in the background.",
        })


def _err(message: str) -> str:
    """Tool-error contract: a JSON string (never a raised exception)."""
    return json.dumps({"error": message})


def register(ctx) -> None:
    """Hermes plugin entry point."""
    ctx.register_memory_provider(BrainDBMemoryProvider())
