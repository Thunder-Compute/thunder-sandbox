"""Remote process wrappers backed by AsyncSSH channels."""

from __future__ import annotations

import asyncio
import uuid
from typing import Generic, TypeVar, cast

import asyncssh

from .._common.exceptions import ConnectionError

T = TypeVar("T", str, bytes)


class _ProcessWriter(Generic[T]):
    """Small transport-neutral wrapper around an AsyncSSH stdin writer."""

    def __init__(self, stream: asyncssh.SSHWriter[T]) -> None:
        self._stream: asyncssh.SSHWriter[T] = stream

    def write(self, data: T) -> int:
        self._stream.write(data)
        return len(data)

    async def drain(self) -> None:
        await self._stream.drain()

    def write_eof(self) -> None:
        self._stream.write_eof()


class _PreservingReader(Generic[T]):
    """Drain an SSH stream without discarding data a caller has not read yet."""

    def __init__(self, stream: object, *, text: bool) -> None:
        self._stream = stream
        self._empty: T = cast(T, "" if text else b"")
        self._newline: T = cast(T, "\n" if text else b"\n")
        self._buffer: T = self._empty
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._eof = False
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            while True:
                chunk = await self._stream.read(65536)  # type: ignore[attr-defined]
                async with self._condition:
                    if not chunk:
                        self._eof = True
                        self._condition.notify_all()
                        return
                    self._buffer += chunk
                    self._condition.notify_all()
        except asyncio.CancelledError:
            async with self._condition:
                self._eof = True
                self._condition.notify_all()
            raise
        except Exception as exc:
            async with self._condition:
                self._error = exc
                self._eof = True
                self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None and not self._buffer:
            raise self._error

    async def read(self, n: int = -1) -> T:
        self.start()
        async with self._condition:
            if n < 0:
                await self._condition.wait_for(lambda: self._eof)
                self._raise_if_failed()
                result, self._buffer = self._buffer, self._empty
                return result
            if n == 0:
                return self._empty
            await self._condition.wait_for(lambda: bool(self._buffer) or self._eof)
            self._raise_if_failed()
            result, self._buffer = self._buffer[:n], self._buffer[n:]
            return result

    async def readline(self) -> T:
        self.start()
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._newline in self._buffer or self._eof
            )
            self._raise_if_failed()
            position = self._buffer.find(self._newline)
            end = position + 1 if position >= 0 else len(self._buffer)
            result, self._buffer = self._buffer[:end], self._buffer[end:]
            return result

    async def wait_eof(self) -> None:
        self.start()
        async with self._condition:
            await self._condition.wait_for(lambda: self._eof)
            if self._error is not None:
                raise self._error

    def __aiter__(self) -> "_PreservingReader[T]":
        return self

    async def __anext__(self) -> T:
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line


class Process(Generic[T]):
    """A remote process backed directly by an AsyncSSH channel."""

    def __init__(
        self,
        process: asyncssh.SSHClientProcess[T],
        *,
        timeout: float | None = None,
        text: bool = True,
    ) -> None:
        self._process: asyncssh.SSHClientProcess[T] = process
        self._timeout = timeout
        # SSH doesn't expose the remote PID. Keep this as an opaque handle,
        # rather than reporting the PID of a local transport process.
        self.id = str(uuid.uuid4())
        self.stdin: _ProcessWriter[T] = _ProcessWriter(process.stdin)
        self.stdout: _PreservingReader[T] = _PreservingReader(
            process.stdout, text=text
        )
        self.stderr: _PreservingReader[T] = _PreservingReader(
            process.stderr, text=text
        )

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def poll(self) -> int | None:
        return self._process.returncode

    async def wait(self, *, timeout: float | None = None) -> int:
        resolved_timeout = self._timeout if timeout is None else timeout
        self.stdout.start()
        self.stderr.start()

        async def wait_and_drain() -> None:
            await self._process.wait_closed()
            await asyncio.gather(self.stdout.wait_eof(), self.stderr.wait_eof())

        await asyncio.wait_for(wait_and_drain(), timeout=resolved_timeout)
        returncode = self._process.returncode
        if returncode is None:
            raise ConnectionError(
                "sandbox SSH process ended without reporting an exit status"
            )
        return returncode

    async def terminate(self) -> None:
        self._process.terminate()


__all__ = ["Process"]
