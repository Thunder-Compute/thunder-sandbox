"""Remote process wrappers backed by the system SSH client."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Generic, IO, TypeVar

T = TypeVar("T", str, bytes)
StreamReader = IO[T]
StreamWriter = IO[T]


class ContainerProcess(Generic[T]):
    def __init__(self, process: subprocess.Popen[T], *, timeout: float | None = None) -> None:
        self._process = process
        self._timeout = timeout
        self.id = str(process.pid)
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self, *, timeout: float | None = None) -> int:
        return self._process.wait(timeout=self._timeout if timeout is None else timeout)

    def terminate(self) -> None:
        self._process.terminate()


class AsyncContainerProcess(Generic[T]):
    def __init__(self, process: ContainerProcess[T]) -> None:
        self._process = process
        self.id = process.id
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def poll(self) -> int | None:
        return self._process.poll()

    async def wait(self, *, timeout: float | None = None) -> int:
        return await asyncio.to_thread(self._process.wait, timeout=timeout)

    async def terminate(self) -> None:
        self._process.terminate()
