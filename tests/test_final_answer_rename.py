"""Edge-case tests for Stage C — `submit_result` → `final_answer` rename + slot pattern.

These are UNIT tests: they import `braindb.agent.*` directly and exercise the
internal contract surface (`FunctionTool.name`, the `_build()` factory's
`StopAtTools` config, the run_state slot lifecycle, run_typed's strict
behaviour). No live LLM, no HTTP — fast and deterministic.

They run alongside the existing integration tests; pytest's session-scoped
`_require_live_api` fixture from `conftest.py` still applies (the suite as a
whole expects a healthy stack), but THESE tests don't actually call the API.

Until Stage C / Layer 1 lands, most assertions here are RED on the
`experimental/structured-output-proper` branch (the rename hasn't happened
yet). After the rename they go green and serve as regression coverage.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from braindb.agent import agent as agent_module
from braindb.agent import run_state
from braindb.agent.schemas import (
    AgentAnswer,
    MaintainerDecision,
    SubagentResult,
    WikiWriteResult,
)
from braindb.agent.tools import (
    submit_answer,
    submit_maintainer,
    submit_subagent,
    submit_wiki,
)


# ------------------------------------------------------------------ #
# Layer 1 — rename surface (FAILS until Stage C / Layer 1 ships)      #
# ------------------------------------------------------------------ #

EXPECTED_FINAL_TOOL_NAME = "final_answer"


@pytest.mark.parametrize(
    "tool",
    [submit_answer, submit_maintainer, submit_wiki, submit_subagent],
    ids=["answer", "maintainer", "wiki", "subagent"],
)
def test_submit_tools_renamed_to_final_answer(tool) -> None:
    """Every typed `submit_*` @function_tool must expose name 'final_answer'
    to the SDK after the rename. The LLM sees this name in the tool catalog;
    a mismatch with the prompt or `StopAtTools` config breaks termination."""
    assert hasattr(tool, "name"), (
        f"{tool!r} is not a FunctionTool — did @function_tool decoration get dropped?"
    )
    assert tool.name == EXPECTED_FINAL_TOOL_NAME, (
        f"{tool!r}.name={tool.name!r}; expected {EXPECTED_FINAL_TOOL_NAME!r} after rename"
    )


def test_stop_at_tools_uses_final_answer() -> None:
    """The `_build()` factory must configure `StopAtTools` with the new name.
    Build all four agents and inspect their tool_use_behavior."""
    agents_to_check = [
        agent_module.get_agent(),
        agent_module.get_maintainer_agent(),
        agent_module.get_writer_agent(),
        agent_module.get_subagent(),
    ]
    for a in agents_to_check:
        beh = a.tool_use_behavior
        # SDK stores it as a dict {"stop_at_tool_names": [...]} OR as a
        # StopAtTools dataclass with the same attribute. Accept both shapes.
        names = (
            beh.get("stop_at_tool_names") if isinstance(beh, dict)
            else getattr(beh, "stop_at_tool_names", None) or getattr(beh, "tool_names", None)
        )
        assert names is not None, f"{a.name}: tool_use_behavior {beh!r} has no recognisable stop-names"
        assert EXPECTED_FINAL_TOOL_NAME in names, (
            f"{a.name}: StopAtTools={names!r}; expected to include {EXPECTED_FINAL_TOOL_NAME!r}"
        )


@pytest.mark.parametrize(
    "prompt_path",
    [
        Path("braindb/agent/prompts/system_prompt.md"),
        Path("braindb/agent/prompts/wiki_maintainer_prompt.md"),
        Path("braindb/agent/prompts/wiki_writer_prompt.md"),
    ],
    ids=["system", "wiki_maintainer", "wiki_writer"],
)
def test_prompts_no_stale_submit_result(prompt_path: Path) -> None:
    """Prompt files must NOT contain the literal `submit_result` after the
    rename — otherwise the LLM gets a confused contract (catalog says
    `final_answer`, prompt says `submit_result`)."""
    repo_root = Path(__file__).parent.parent  # tests/ → repo root
    full = repo_root / prompt_path
    assert full.exists(), f"prompt missing: {full}"
    body = full.read_text(encoding="utf-8")
    assert "submit_result" not in body, (
        f"{prompt_path} still references 'submit_result' — should be 'final_answer'"
    )


# ------------------------------------------------------------------ #
# Slot pattern (already shipped in 8560cfa; regression coverage)      #
# ------------------------------------------------------------------ #


def test_slot_install_and_release_isolation() -> None:
    """Two sequential install/release cycles produce distinct slot objects.
    Within a cycle, `record_submit` mutates the active slot; after release,
    the outer slot's value is unchanged."""
    slot1, token1 = run_state.install_slot()
    assert slot1.value is None
    run_state.record_submit("payload-1")
    assert slot1.value == "payload-1"
    run_state.release_slot(token1)

    slot2, token2 = run_state.install_slot()
    assert slot2 is not slot1
    assert slot2.value is None       # fresh slot, not stale data from slot1
    run_state.record_submit("payload-2")
    assert slot2.value == "payload-2"
    assert slot1.value == "payload-1"  # the released slot still holds its old data, but is no longer the ContextVar's value
    run_state.release_slot(token2)


def test_slot_nested_install_release() -> None:
    """The wiki maintainer/writer pattern: parent run_typed installs a slot,
    a delegated subagent installs its own, releases, then parent finalises.
    The child's record_submit must NOT contaminate the parent's slot."""
    parent_slot, parent_token = run_state.install_slot()
    run_state.record_submit("parent-data")
    assert parent_slot.value == "parent-data"

    # Child run_typed enters
    child_slot, child_token = run_state.install_slot()
    assert child_slot is not parent_slot
    assert child_slot.value is None
    run_state.record_submit("child-data")
    assert child_slot.value == "child-data"
    assert parent_slot.value == "parent-data"  # unaffected
    run_state.release_slot(child_token)

    # Back in parent context; record_submit should target parent again
    run_state.record_submit("parent-data-after-child")
    assert parent_slot.value == "parent-data-after-child"
    run_state.release_slot(parent_token)


def test_record_submit_outside_run_is_silent_noop() -> None:
    """If `record_submit` is called outside any `install_slot()` scope (e.g.
    a bug in a tool, or stale state), it must NOT raise. The current
    implementation silently drops the payload because the ContextVar
    defaults to None."""
    # This must not raise even with no active slot.
    run_state.record_submit("orphan-payload")
    # The slot var should still be None
    assert run_state._slot_var.get() is None


# ------------------------------------------------------------------ #
# run_typed strict-mode behaviour                                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_run_typed_raises_when_submit_never_fires() -> None:
    """If Runner.run completes without any `submit_*` having called
    record_submit, run_typed must raise RuntimeError — the strict-mode
    invariant. Surfaces 'model emitted prose' / 'max_turns exhausted'
    as a real failure rather than silently returning bad data."""
    fake_agent = mock.MagicMock(name="fake_agent")
    fake_agent.name = "FakeAgent"

    async def fake_runner_run(starting_agent, input, max_turns, **kwargs):
        # Pretend the LLM ran but never called any submit_*.
        return mock.MagicMock(final_output="some-prose-text")

    with mock.patch.object(agent_module.Runner, "run", new=fake_runner_run):
        with pytest.raises(RuntimeError, match="did not call final_answer|did not submit"):
            await agent_module.run_typed("query", fake_agent, AgentAnswer, max_turns=5)


@pytest.mark.asyncio
async def test_run_typed_returns_typed_payload_when_submitted() -> None:
    """If record_submit IS called during Runner.run with the expected typed
    payload, run_typed returns that exact instance — the typed-final
    contract."""
    fake_agent = mock.MagicMock(name="fake_agent")
    fake_agent.name = "FakeAgent"
    expected = AgentAnswer(answer="hello world")

    async def fake_runner_run(starting_agent, input, max_turns, **kwargs):
        # Simulate a submit_* tool body firing during the run
        run_state.record_submit(expected)
        return mock.MagicMock(final_output="ok")

    with mock.patch.object(agent_module.Runner, "run", new=fake_runner_run):
        got = await agent_module.run_typed("query", fake_agent, AgentAnswer, max_turns=5)
    assert got is expected
    assert got.answer == "hello world"


# ------------------------------------------------------------------ #
# Pydantic typed-arg validation (regression cover)                     #
# ------------------------------------------------------------------ #


def test_typed_models_validate_strictly() -> None:
    """The @function_tool argument schemas are derived from these Pydantic
    models. Validation MUST reject malformed input — that's what protects
    the typed-final contract from the LLM emitting garbage args."""
    # Each model has at least one required field; passing the wrong shape
    # must raise pydantic.ValidationError.
    with pytest.raises(Exception):  # pydantic.ValidationError
        AgentAnswer(answer=123)  # wrong type
    with pytest.raises(Exception):
        MaintainerDecision()  # missing 'action'
    with pytest.raises(Exception):
        WikiWriteResult()  # missing 'mode' and 'body'
    with pytest.raises(Exception):
        SubagentResult()  # missing 'result'
    # Round-trip a valid one to confirm the happy path still works.
    a = AgentAnswer(answer="x")
    assert a.answer == "x"
