"""
Per-run side-channel for the agent's final structured payload.

Why this exists: `Agent(output_type=<PydanticModel>)` makes the SDK pass
`response_format: json_schema` on EVERY LLM call (not just the final
one), which steers weaker models to satisfy the schema on turn 1 and
skip tools entirely. We therefore build agents WITHOUT `output_type` so
intermediate turns are free — but then `StopAtTools` would `str()`-coerce
the stop-tool's return into `result.final_output`, and we'd lose the
typed instance.

This module is the bridge: each `submit_*` tool body parks the
SDK-validated payload via `record_submit(payload)`; `run_typed` reads it
back via `slot.value` after `Runner.run` returns.

## Why a mutable slot, not just `ContextVar[Any]`

ContextVar values are inherited by reference into child asyncio Tasks,
but `.set()` inside a child Task does NOT propagate up to the parent.
The openai-agents SDK runs tool bodies (including parallel-tool batches)
inside such child Tasks, so a naive `last_submit.set(payload)` in the
tool body is invisible to the surrounding `run_typed`. Putting a mutable
container in the ContextVar instead — and mutating its `.value` from the
tool — works across that boundary because every Task sees the same
object reference. The standard `set(slot) + reset(token)` lifecycle in
`run_typed` keeps nested runs (parent → `delegate_to_subagent` →
subagent) isolated: each level uses its own `_Slot`.
"""
from contextvars import ContextVar
from typing import Any


class _Slot:
    """One-shot holder for the validated payload of a single agent run."""
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: Any = None


# Default None — `run_typed` always installs its own slot before awaiting
# `Runner.run`. A `None` here at submit time means "called outside a
# run_typed scope" and is just silently dropped (no slot to write to).
_slot_var: ContextVar["_Slot | None"] = ContextVar(
    "braindb_last_submit_slot", default=None,
)


def install_slot() -> tuple[_Slot, object]:
    """Used by `run_typed` to start a run. Returns `(slot, token)`; pass
    `token` to `release_slot` in a `finally:` to restore the previous
    context (so nested runs are isolated)."""
    slot = _Slot()
    token = _slot_var.set(slot)
    return slot, token


def release_slot(token: object) -> None:
    """Restore the previous slot (call in `finally:` after `install_slot`)."""
    _slot_var.reset(token)  # type: ignore[arg-type]


def record_submit(payload: Any) -> None:
    """Called from inside every `submit_*` tool body. The SDK has already
    validated `payload` against the tool's Pydantic argument schema, so
    the value parked here is the typed final answer by construction.

    Mutates the slot in place (does NOT call `ContextVar.set(...)`) — see
    module docstring for why."""
    slot = _slot_var.get()
    if slot is not None:
        slot.value = payload
