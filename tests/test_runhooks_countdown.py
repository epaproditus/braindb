"""Edge-case tests for Stage C / Layer 3 — RunHooks countdown nudge.

The contract being tested:

- A `CountdownHooks` class lives in `braindb.agent.hooks` and subclasses
  `agents.RunHooks`. It implements `on_llm_start`, counting LLM turns and,
  when ≤ `threshold` turns remain before `max_turns`, mutating the
  `input_items` list passed to the LLM to APPEND a synthetic nudge
  reminding the model to finalise via `final_answer`.

- The nudge fires at most ONCE per run (idempotent). After firing, the
  hook does not re-inject on subsequent turns.

- The hook is defensive: a malformed `input_items` argument or any
  unexpected SDK shape change must not crash the run — exceptions are
  swallowed (and logged) so the agent loop keeps going.

- `threshold=0` disables the hook (safety hatch / opt-out).

- `max_turns < threshold` (weird config) does not crash; behaves as
  "always at threshold from turn 1" but still only fires once.

These tests instantiate the hook directly and call `on_llm_start`
synchronously via asyncio — no live LLM, no real agent loop.
"""
from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from braindb.agent.hooks import CountdownHooks

EXPECTED_TOOL_NAME = "final_answer"


def _run(coro):
    """Run a single coroutine to completion. Each test gets a fresh loop."""
    return asyncio.get_event_loop().run_until_complete(coro) if not asyncio.iscoroutine(coro) else asyncio.run(coro)


def _make_args(input_items: list | None = None):
    """Helper to build the args `on_llm_start` is called with. We only care
    about `input_items` (the mutable list the hook may append to); the other
    args are stubs."""
    ctx = mock.MagicMock(name="context")
    agent = mock.MagicMock(name="agent", spec=[])
    agent.name = "TestAgent"
    return ctx, agent, "system-prompt-stub", (input_items if input_items is not None else [])


@pytest.mark.asyncio
async def test_countdown_idle_when_far_from_max() -> None:
    """If we're nowhere near max_turns - threshold, the hook must not
    inject anything into input_items."""
    hooks = CountdownHooks(max_turns=20, threshold=5, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    for _ in range(3):  # 3 LLM calls, well below max_turns - threshold = 15
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    assert items == [], f"hook fired too early; items={items!r}"
    assert hooks._fired is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_countdown_fires_at_threshold() -> None:
    """When the running turn count crosses `max_turns - threshold`, the
    hook must append exactly one item to `input_items` and flip its
    fired flag."""
    max_turns, threshold = 20, 5
    hooks = CountdownHooks(max_turns=max_turns, threshold=threshold, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    # Turns 1..(max_turns - threshold - 1) must NOT fire.
    for i in range(max_turns - threshold - 1):
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    assert items == []
    # The next call crosses the threshold → fires.
    ctx, agent, sp, _ = _make_args(items)
    await hooks.on_llm_start(ctx, agent, sp, items)
    assert len(items) == 1, f"expected exactly 1 nudge appended, got {items!r}"
    nudge = items[0]
    # The nudge must mention the final-tool name; format can be dict or str.
    nudge_text = nudge.get("content") if isinstance(nudge, dict) else str(nudge)
    assert EXPECTED_TOOL_NAME in nudge_text, f"nudge missing tool name; got {nudge_text!r}"
    assert hooks._fired is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_countdown_idempotent_after_firing() -> None:
    """Once the hook has injected, subsequent on_llm_start calls must not
    add more nudges to input_items (the prior nudge is already in the
    conversation; duplicating is spam)."""
    hooks = CountdownHooks(max_turns=10, threshold=3, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    # Push past the threshold to force firing
    for _ in range(8):
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    assert hooks._fired is True  # type: ignore[attr-defined]
    nudges_after_first = len(items)
    # Several more turns — should not append again
    for _ in range(5):
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    assert len(items) == nudges_after_first, "hook re-injected on subsequent turns"


@pytest.mark.asyncio
async def test_countdown_disabled_when_threshold_zero() -> None:
    """`threshold=0` disables the hook entirely — opt-out for ops who don't
    want the nudge."""
    hooks = CountdownHooks(max_turns=10, threshold=0, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    for _ in range(50):  # Way past any reasonable max_turns
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    assert items == [], "hook fired despite threshold=0"
    assert hooks._fired is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_countdown_max_turns_below_threshold_safe() -> None:
    """Pathological config (`max_turns=3, threshold=5`) must NOT crash.
    The hook should still fire at most once and not blow up."""
    hooks = CountdownHooks(max_turns=3, threshold=5, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    for _ in range(5):
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    # The exact when-fires policy is implementation-defined; the contract is:
    # at most one nudge, no exception raised.
    assert len(items) <= 1


@pytest.mark.asyncio
async def test_countdown_does_not_break_normal_completion() -> None:
    """If the model finalises BEFORE the threshold is hit, the hook should
    not have injected anything (record-of-non-action: nothing in items)."""
    hooks = CountdownHooks(max_turns=20, threshold=5, tool_name=EXPECTED_TOOL_NAME)
    items: list = []
    # Simulate a quick agent that uses 3 turns and submits.
    for _ in range(3):
        ctx, agent, sp, _ = _make_args(items)
        await hooks.on_llm_start(ctx, agent, sp, items)
    # No further LLM calls (agent finished). Items still empty.
    assert items == []
    assert hooks._fired is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_hook_exception_does_not_kill_run() -> None:
    """Internal hook errors (e.g. SDK shape change) must be SWALLOWED so
    the agent loop can keep running. Otherwise a defensive bug in the
    hook brings down production runs."""
    hooks = CountdownHooks(max_turns=20, threshold=5, tool_name=EXPECTED_TOOL_NAME)
    items: list = []

    # Patch the internal `_maybe_inject` to blow up. The public
    # `on_llm_start` must still complete without raising.
    with mock.patch.object(hooks, "_maybe_inject", side_effect=RuntimeError("sim shape change")):
        ctx, agent, sp, _ = _make_args(items)
        try:
            await hooks.on_llm_start(ctx, agent, sp, items)
        except Exception as e:  # noqa: BLE001 — that's the point
            pytest.fail(f"on_llm_start let an exception escape: {e!r}")
