"""The canonical lifecycle of an owned, temporary sandbox."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .._common.lifecycle import finish_cleanup
from .._common.types import SandboxInfo

if TYPE_CHECKING:
    from .sandbox import Sandbox


@asynccontextmanager
async def ephemeral(
    sandbox: Sandbox, *, ready_timeout: float, cleanup_timeout: float,
    on_status: Callable[[SandboxInfo], None] | None,
) -> AsyncIterator[Sandbox]:
    failure: BaseException | None = None
    try:
        await sandbox.wait_until_ready(timeout=ready_timeout, on_status=on_status)
        yield sandbox
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            await finish_cleanup(asyncio.wait_for(
                sandbox.terminate(timeout=cleanup_timeout), timeout=cleanup_timeout,
            ))
        except BaseException as cleanup_error:
            if failure is not None:
                raise failure from cleanup_error
            raise
