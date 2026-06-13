"""
External-integration endpoints — consumed by the Hermes memory provider (see
`integrations/hermes/`) and similar external clients. These are NOT used by
BrainDB's own app, agent, or watcher; they exist purely so an outside agent can
(a) self-configure from BrainDB's shipped skill text (single source of truth, no
copied prompt) and (b) hand a file to BrainDB's existing ingestion pipeline over
HTTP without needing a shared filesystem.

Kept isolated in this dedicated router (and registered with a single
`include_router` line in main.py) so the integration surface is obvious and the
core routers stay untouched. Everything here is additive — no existing route,
schema, or behaviour is affected.
"""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api/v1", tags=["integrations"])

# Mirror the watcher's watch dir + allow-list so an upload lands exactly where
# the existing ingestion pipeline (braindb/ingest_watcher.py) already looks.
_WATCH_DIR = Path(os.getenv("INGEST_WATCH_DIR", "data/sources"))
_ALLOWED_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".html", ".xml"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — generous for text docs, blocks abuse
_SKILLS_DIR = Path("skills")


def _safe_basename(raw: str) -> str:
    """Reduce a client-supplied name to a bare filename — no path traversal.

    We only ever join the basename to a fixed directory, so traversal is
    impossible even before this check; the explicit guard just rejects obvious
    junk early with a clear 400.
    """
    base = os.path.basename((raw or "").replace("\\", "/").strip())
    if not base or base in (".", "..") or base.startswith("."):
        raise HTTPException(400, "Invalid or unsafe name")
    return base


@router.get("/skill/{name}", response_class=PlainTextResponse)
def get_skill(name: str) -> str:
    """Serve a shipped skill's markdown (e.g. `name=braindb-agent`).

    Lets an external client load its usage instructions live from BrainDB rather
    than bundling a copy that could drift. Read-only.
    """
    safe = _safe_basename(name)
    path = _SKILLS_DIR / safe / "SKILL.md"
    if not path.is_file():
        raise HTTPException(404, f"No skill named {safe!r}")
    return path.read_text(encoding="utf-8")


@router.post("/entities/datasources/upload", status_code=202)
async def upload_datasource(request: Request, filename: str):
    """Accept a file (raw request body) and drop it into `data/sources/` so the
    existing watcher ingests + fact-extracts it.

    Adds NO ingestion logic of its own — it only lands the bytes where the
    pipeline already polls. Extraction is asynchronous (the watcher picks the
    file up on its next tick). Raw body + `filename` query param is used
    deliberately so this needs no multipart dependency.
    """
    fname = _safe_basename(filename)
    ext = Path(fname).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported extension {ext!r}; allowed: {sorted(_ALLOWED_EXTS)}")

    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty body")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (> {_MAX_UPLOAD_BYTES} bytes)")

    _WATCH_DIR.mkdir(parents=True, exist_ok=True)
    dest = _WATCH_DIR / fname
    # No-overwrite on name collision (mirrors the watcher's own dedup).
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 1
        while (cand := _WATCH_DIR / f"{stem}.{i}{suffix}").exists():
            i += 1
        dest = cand
    dest.write_bytes(data)

    return {
        "filename": dest.name,
        "status": "accepted",
        "message": "Uploaded to data/sources/; the watcher will ingest and fact-extract it shortly.",
    }
