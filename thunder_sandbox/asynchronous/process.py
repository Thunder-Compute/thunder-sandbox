"""Remote process handles, backed by an execution channel."""

from __future__ import annotations

import asyncio
import codecs
import io
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Generic, TypeVar, cast

if TYPE_CHECKING:
    from .._channels._base import ExecHandle
    from .._channels._spec import ExecStatus

T = TypeVar("T", str, bytes)


class _UnsupportedWriter(Generic[T]):
    """stdin for a detached job is intentionally not connection-backed."""

    @staticmethod
    def _unsupported() -> io.UnsupportedOperation:
        return io.UnsupportedOperation(
            "stdin is unavailable for durable commands; use files, arguments, "
            "environment variables, or pty=True"
        )

    def write(self, data: T) -> int:
        raise self._unsupported()

    async def drain(self) -> None:
        raise self._unsupported()

    def write_eof(self) -> None:
        raise self._unsupported()


class _HandleReader(Generic[T]):
    """Reads one stream of an execution-channel process.

    The channel delivers raw chunks; buffering, line splitting and text
    decoding happen here, the same way for every channel.
    """

    def __init__(self, chunks: Callable[[], AsyncIterator[bytes]], *, text: bool) -> None:
        self._open_chunks = chunks
        self._chunks: AsyncIterator[bytes] | None = None
        self._empty: T = cast(T, "" if text else b"")
        self._newline: T = cast(T, "\n" if text else b"\n")
        self._buffer: T = self._empty
        self._decoder = codecs.getincrementaldecoder("utf-8")() if text else None
        self._eof = False
        self._lock = asyncio.Lock()

    def start(self) -> None:
        return None

    async def _fill(self) -> None:
        if self._chunks is None:
            self._chunks = self._open_chunks()
        try:
            chunk = await self._chunks.__anext__()
        except StopAsyncIteration:
            if self._decoder is not None:
                self._buffer += cast(T, self._decoder.decode(b"", final=True))
            self._eof = True
            return
        value = self._decoder.decode(chunk, final=False) if self._decoder is not None else chunk
        self._buffer += cast(T, value)

    async def read(self, n: int = -1) -> T:
        if n == 0:
            return self._empty
        async with self._lock:
            if n < 0:
                while not self._eof:
                    await self._fill()
                result, self._buffer = self._buffer, self._empty
                return result
            while not self._buffer and not self._eof:
                await self._fill()
            result, self._buffer = self._buffer[:n], self._buffer[n:]
            return result

    async def readline(self) -> T:
        async with self._lock:
            while self._newline not in self._buffer and not self._eof:
                await self._fill()
            position = self._buffer.find(self._newline)
            end = position + 1 if position >= 0 else len(self._buffer)
            result, self._buffer = self._buffer[:end], self._buffer[end:]
            return result

    async def wait_eof(self) -> None:
        async with self._lock:
            while not self._eof:
                await self._fill()

    def __aiter__(self) -> "_HandleReader[T]":
        return self

    async def __anext__(self) -> T:
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line


class _HandleWriter(Generic[T]):
    """stdin of an execution-channel process, with the asyncio writer's shape:
    ``write`` and ``write_eof`` queue, ``drain`` waits until the process has
    taken everything queued."""

    _EOF = object()

    def __init__(self, handle: "ExecHandle", *, text: bool) -> None:
        self._handle = handle
        self._text = text
        self._queue: list[object] = []
        self._worker: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        while self._queue:
            item = self._queue.pop(0)
            try:
                if item is self._EOF:
                    await self._handle.close_stdin()
                else:
                    await self._handle.write_stdin(cast(bytes, item))
            except BaseException as exc:
                self._error = exc
                self._queue.clear()
                return

    def write(self, data: T) -> int:
        self._raise_if_failed()
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self._queue.append(payload)
        self._start()
        return len(data)

    def write_eof(self) -> None:
        self._raise_if_failed()
        self._queue.append(self._EOF)
        self._start()

    async def drain(self) -> None:
        if self._worker is not None:
            await asyncio.wait({self._worker})
        self._raise_if_failed()


class Process(Generic[T]):
    """A remote process: a reconnectable durable job, or a PTY session.

    Its work is done by an execution channel's handle; this class gives it the
    asyncio-like shape of ``stdout``, ``stderr``, ``stdin``, ``wait`` and
    ``returncode``.
    """

    def __init__(
        self, handle: "ExecHandle", *, timeout: float | None, text: bool, durable: bool
    ) -> None:
        self._handle = handle
        self._timeout = timeout
        self._durable = durable
        self._status: "ExecStatus | None" = None
        self.id = handle.exec_id
        self.stdin: _UnsupportedWriter[T] | _HandleWriter[T] = (
            _UnsupportedWriter() if durable else _HandleWriter(handle, text=text)
        )
        self.stdout: _HandleReader[T] = _HandleReader(handle.stdout, text=text)
        self.stderr: _HandleReader[T] = _HandleReader(handle.stderr, text=text)

    @classmethod
    def from_handle(
        cls,
        handle: "ExecHandle",
        *,
        timeout: float | None,
        text: bool,
        durable: bool,
    ) -> "Process[str] | Process[bytes]":
        """A process run by an execution channel."""
        return cast("Process[str] | Process[bytes]", cls(
            handle, timeout=timeout, text=text, durable=durable
        ))

    def _record(self, status: "ExecStatus") -> int | None:
        self._status = status
        return self.returncode

    @property
    def returncode(self) -> int | None:
        status = self._handle.last_status() or self._status
        if status is None or status.reason is None:
            return None
        # A process whose end nothing observed reads as killed, as before.
        return status.returncode if status.returncode is not None else 137

    @property
    def is_durable(self) -> bool:
        """Whether this handle can reconnect independently of its SSH channel."""
        return self._durable

    async def poll(self) -> int | None:
        return self._record(await self._handle.poll())

    async def wait(self, *, timeout: float | None = None) -> int:
        resolved = self._timeout if timeout is None else timeout
        status = await asyncio.wait_for(self._handle.wait(), timeout=resolved)
        code = self._record(status)
        assert code is not None
        return code

    async def cleanup(self) -> None:
        """Wait for a durable job, then remove all of its remote artifacts."""
        await self._handle.cleanup()

    async def terminate(self) -> None:
        self._record(await self._handle.terminate())


__all__ = ["Process"]
