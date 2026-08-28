"""Remote process wrappers backed by AsyncSSH channels."""

from __future__ import annotations

import asyncio
import uuid
from typing import Generic, TypeVar

import asyncssh

T = TypeVar("T", str, bytes)


class Process(Generic[T]):
    """A remote process backed directly by an AsyncSSH channel."""

    def __init__(
        self,
        process: asyncssh.SSHClientProcess[T],
        *,
        timeout: float | None = None,
    ) -> None:
        self._process = process
        self._timeout = timeout
        # SSH doesn't expose the remote PID. Keep this as an opaque handle,
        # rather than reporting the PID of a local transport process.
        self.id = str(uuid.uuid4())
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def poll(self) -> int | None:
        return self._process.returncode

    async def wait(self, *, timeout: float | None = None) -> int:
        resolved_timeout = self._timeout if timeout is None else timeout
        await asyncio.wait_for(
            self._process.wait_closed(), timeout=resolved_timeout
        )
        return self._process.returncode

    async def terminate(self) -> None:
        self._process.terminate()


__all__ = ["Process"]
