"""
External-integration endpoints (the Hermes memory-provider surface):
  GET  /api/v1/skill/{name}
  POST /api/v1/entities/datasources/upload

Runs against the isolated test stack (the `api` fixture). That stack has NO
watcher and uses the throwaway ./data_test mount, so uploads here never trigger
ingestion (no LLM) and never touch the personal stack.
"""
import requests

SKILL = "/api/v1/skill"
UPLOAD = "/api/v1/entities/datasources/upload"


def test_skill_served(api):
    r = requests.get(f"{api}{SKILL}/braindb-agent", timeout=10)
    assert r.status_code == 200
    assert r.text.strip()  # non-empty markdown body


def test_skill_unknown_404(api):
    r = requests.get(f"{api}{SKILL}/no-such-skill", timeout=10)
    assert r.status_code == 404


def test_skill_unsafe_name_400(api):
    # leading dot -> rejected by _safe_basename before any disk access
    r = requests.get(f"{api}{SKILL}/.hidden", timeout=10)
    assert r.status_code == 400


def test_upload_accepted(api, test_tag):
    name = f"{test_tag}.md"
    r = requests.post(
        f"{api}{UPLOAD}",
        params={"filename": name},
        data=b"# note\n\nbody text",
        headers={"Content-Type": "application/octet-stream"},
        timeout=10,
    )
    assert r.status_code == 202, r.text
    assert r.json()["filename"] == name


def test_upload_bad_extension_400(api, test_tag):
    r = requests.post(
        f"{api}{UPLOAD}", params={"filename": f"{test_tag}.exe"}, data=b"x", timeout=10
    )
    assert r.status_code == 400


def test_upload_empty_400(api, test_tag):
    r = requests.post(
        f"{api}{UPLOAD}", params={"filename": f"{test_tag}.md"}, data=b"", timeout=10
    )
    assert r.status_code == 400


def test_upload_collision_no_overwrite(api, test_tag):
    name = f"{test_tag}_c.md"
    first = requests.post(f"{api}{UPLOAD}", params={"filename": name}, data=b"one", timeout=10)
    second = requests.post(f"{api}{UPLOAD}", params={"filename": name}, data=b"two", timeout=10)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["filename"] == name
    assert second.json()["filename"] == f"{test_tag}_c.1.md"
