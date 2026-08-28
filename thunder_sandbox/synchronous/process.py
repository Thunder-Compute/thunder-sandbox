"""Blocking adapters over native asynchronous remote processes."""

from __future__ import annotations

from typing import Generic, TypeVar

from ._bridge import AsyncBridge
from ..asynchronous.process import Process as NativeProcess

T = TypeVar("T", str, bytes)


class StreamReader(Generic[T]):
    def __init__(self, bridge: AsyncBridge, stream: object) -> None:
        self._bridge = bridge
        self._stream = stream

    def read(self, n: int = -1) -> T:
        return self._bridge.run(self._stream.read(n))  # type: ignore[attr-defined, no-any-return]

    def readline(self) -> T:
        return self._bridge.run(self._stream.readline())  # type: ignore[attr-defined, no-any-return]

    async def read_async(self, n: int = -1) -> T:
        return await self._bridge.run_async(self._stream.read(n))  # type: ignore[attr-defined, no-any-return]

    async def readline_async(self) -> T:
        return await self._bridge.run_async(self._stream.readline())  # type: ignore[attr-defined, no-any-return]

    def __iter__(self) -> "StreamReader[T]":
        return self

    def __next__(self) -> T:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


class StreamWriter(Generic[T]):
    def __init__(self, bridge: AsyncBridge, stream: object) -> None:
        self._bridge = bridge
        self._stream = stream

    def write(self, data: T) -> int:
        async def write() -> int:
            self._stream.write(data)  # type: ignore[attr-defined]
            await self._stream.drain()  # type: ignore[attr-defined]
            return len(data)

        return self._bridge.run(write())

    def flush(self) -> None:
        self._bridge.run(self._stream.drain())  # type: ignore[attr-defined]

    async def write_async(self, data: T) -> int:
        async def write() -> int:
            self._stream.write(data)  # type: ignore[attr-defined]
            await self._stream.drain()  # type: ignore[attr-defined]
            return len(data)

        return await self._bridge.run_async(write())

    async def flush_async(self) -> None:
        await self._bridge.run_async(self._stream.drain())  # type: ignore[attr-defined]

    def close(self) -> None:
        async def close() -> None:
            self._stream.write_eof()  # type: ignore[attr-defined]
            await self._stream.drain()  # type: ignore[attr-defined]

        self._bridge.run(close())

    async def close_async(self) -> None:
        async def close() -> None:
            self._stream.write_eof()  # type: ignore[attr-defined]
            await self._stream.drain()  # type: ignore[attr-defined]

        await self._bridge.run_async(close())


class Process(Generic[T]):
    def __init__(
        self, bridge: AsyncBridge, process: NativeProcess[T]
    ) -> None:
        self._bridge = bridge
        self._process = process
        self.id = process.id
        self.stdin = StreamWriter[T](bridge, process.stdin)
        self.stdout = StreamReader[T](bridge, process.stdout)
        self.stderr = StreamReader[T](bridge, process.stderr)

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def poll(self) -> int | None:
        return self._bridge.run(self._process.poll())

    async def poll_async(self) -> int | None:
        return await self._bridge.run_async(self._process.poll())

    def wait(self, *, timeout: float | None = None) -> int:
        return self._bridge.run(self._process.wait(timeout=timeout))

    async def wait_async(self, *, timeout: float | None = None) -> int:
        return await self._bridge.run_async(self._process.wait(timeout=timeout))

    def terminate(self) -> None:
        self._bridge.run(self._process.terminate())

    async def terminate_async(self) -> None:
        await self._bridge.run_async(self._process.terminate())


__all__ = ["Process"]
