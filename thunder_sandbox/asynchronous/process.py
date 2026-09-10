"""Attached and durable remote process handles."""

from __future__ import annotations

import asyncio
import codecs
import io
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar, cast

import asyncssh

from .._common.exceptions import ConnectionError, SandboxFailedError
from ._jobs import JobState, JobStatus, can_transition

T = TypeVar("T", str, bytes)
StatusReader = Callable[[float | None], Awaitable[JobStatus]]
OutputReader = Callable[[str, int, int], Awaitable[bytes]]
CleanupJob = Callable[[], Awaitable[None]]
SignalJob = Callable[[str, int], Awaitable[JobStatus]]

PROCESS_POLL_RETRY_SECONDS = 5.0
PROCESS_STATUS_INITIAL_DELAY_SECONDS = 0.1
PROCESS_STATUS_MAX_DELAY_SECONDS = 1.0
PROCESS_TERMINATION_GRACE_SECONDS = 5.0


class _ProcessWriter(Generic[T]):
    """Small transport-neutral wrapper around an AsyncSSH stdin writer."""

    def __init__(self, stream: asyncssh.SSHWriter[T]) -> None:
        self._stream: asyncssh.SSHWriter[T] = stream

    def write(self, data: T) -> int:
        try:
            self._stream.write(data)
        except (OSError, asyncssh.Error) as exc:
            raise _attached_connection_error(exc) from exc
        return len(data)

    async def drain(self) -> None:
        try:
            await self._stream.drain()
        except (OSError, asyncssh.Error) as exc:
            raise _attached_connection_error(exc) from exc

    def write_eof(self) -> None:
        try:
            self._stream.write_eof()
        except (OSError, asyncssh.Error) as exc:
            raise _attached_connection_error(exc) from exc


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
        except (OSError, asyncssh.Error) as exc:
            async with self._condition:
                self._error = _attached_connection_error(exc)
                self._eof = True
                self._condition.notify_all()
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


class _DurableReader(Generic[T]):
    """A reconnectable remote-file reader with a client-side byte cursor."""

    def __init__(
        self,
        stream: str,
        *,
        text: bool,
        read_output: OutputReader,
        terminal: Callable[[], Awaitable[bool]],
        reached_eof: Callable[[str], Awaitable[None]],
        captured: bool,
    ) -> None:
        self._stream = stream
        self._text = text
        self._read_output = read_output
        self._terminal = terminal
        self._reached_eof = reached_eof
        self._empty: T = cast(T, "" if text else b"")
        self._newline: T = cast(T, "\n" if text else b"\n")
        self._buffer: T = self._empty
        self._decoder = codecs.getincrementaldecoder("utf-8")() if text else None
        self._offset = 0
        self._eof = not captured
        self._eof_reported = False
        self._lock = asyncio.Lock()

    def start(self) -> None:
        # Durable output lives in a file, so no channel needs eager draining.
        return None

    @property
    def eof(self) -> bool:
        return self._eof

    async def _report_eof(self) -> None:
        if self._eof_reported:
            return
        self._eof_reported = True
        await self._reached_eof(self._stream)

    async def _fetch(self) -> None:
        if self._eof:
            await self._report_eof()
            return
        # EOF is conclusive only when the job was already terminal before the
        # corresponding file read began. If it exits during either round trip,
        # the next fetch samples terminal state and reads once more, preserving
        # bytes flushed at process exit.
        terminal_before_read = await self._terminal()
        chunk = await self._read_output(self._stream, self._offset, 65536)
        if chunk:
            # The remote offset advances only after the complete chunk has
            # reached client memory. A lost SFTP operation retries this offset.
            self._offset += len(chunk)
            value: str | bytes
            if self._decoder is not None:
                value = self._decoder.decode(chunk, final=False)
            else:
                value = chunk
            self._buffer += cast(T, value)
            return
        if not terminal_before_read:
            await asyncio.sleep(PROCESS_STATUS_INITIAL_DELAY_SECONDS)
            return
        if self._decoder is not None:
            self._buffer += cast(T, self._decoder.decode(b"", final=True))
        self._eof = True
        await self._report_eof()

    async def read(self, n: int = -1) -> T:
        if n == 0:
            return self._empty
        async with self._lock:
            if self._eof:
                await self._report_eof()
            if n < 0:
                while not self._eof:
                    await self._fetch()
                result, self._buffer = self._buffer, self._empty
                return result
            while not self._buffer and not self._eof:
                await self._fetch()
            result, self._buffer = self._buffer[:n], self._buffer[n:]
            return result

    async def readline(self) -> T:
        async with self._lock:
            if self._eof:
                await self._report_eof()
            while self._newline not in self._buffer and not self._eof:
                await self._fetch()
            position = self._buffer.find(self._newline)
            end = position + 1 if position >= 0 else len(self._buffer)
            result, self._buffer = self._buffer[:end], self._buffer[end:]
            return result

    async def wait_eof(self) -> None:
        async with self._lock:
            while not self._eof:
                await self._fetch()
            await self._report_eof()

    def __aiter__(self) -> "_DurableReader[T]":
        return self

    async def __anext__(self) -> T:
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line


class Process(Generic[T]):
    """A remote process backed by a durable job or attached SSH channel."""

    def __init__(
        self,
        process: asyncssh.SSHClientProcess[T],
        *,
        timeout: float | None = None,
        text: bool = True,
    ) -> None:
        self._process: asyncssh.SSHClientProcess[T] | None = process
        self._timeout = timeout
        self._status_reader: StatusReader | None = None
        self._status: JobStatus | None = None
        self._status_lock = asyncio.Lock()
        self._cleanup_job: CleanupJob | None = None
        self._signal_job: SignalJob | None = None
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._termination_lock = asyncio.Lock()
        self._cleaned = False
        self._retain = True
        # SSH doesn't expose the remote PID. Keep this as an opaque handle,
        # rather than reporting the PID of a local transport process.
        self.id = str(uuid.uuid4())
        self.stdin: _ProcessWriter[T] | _UnsupportedWriter[T] = _ProcessWriter(
            process.stdin
        )
        self.stdout: _PreservingReader[T] | _DurableReader[T] = _PreservingReader(
            process.stdout, text=text
        )
        self.stderr: _PreservingReader[T] | _DurableReader[T] = _PreservingReader(
            process.stderr, text=text
        )

    @classmethod
    def durable(
        cls,
        job_id: str,
        *,
        status: JobStatus,
        read_status: StatusReader,
        read_output: OutputReader,
        cleanup_job: CleanupJob,
        signal_job: SignalJob,
        timeout: float | None,
        text: bool,
        stdout: str = "capture",
        stderr: str = "capture",
        retain: bool = False,
    ) -> "Process[str] | Process[bytes]":
        instance = cls.__new__(cls)
        instance._process = None
        instance._timeout = timeout
        instance._status_reader = read_status
        instance._status = status
        instance._status_lock = asyncio.Lock()
        instance._cleanup_job = cleanup_job
        instance._signal_job = signal_job
        instance._cleanup_lock = asyncio.Lock()
        instance._cleanup_task = None
        instance._termination_lock = asyncio.Lock()
        instance._cleaned = False
        instance._retain = retain
        instance.id = job_id
        instance.stdin = _UnsupportedWriter()
        instance.stdout = _DurableReader(
            "stdout",
            text=text,
            read_output=read_output,
            terminal=instance._is_terminal,
            reached_eof=instance._stream_reached_eof,
            captured=stdout == "capture",
        )
        instance.stderr = _DurableReader(
            "stderr",
            text=text,
            read_output=read_output,
            terminal=instance._is_terminal,
            reached_eof=instance._stream_reached_eof,
            captured=stderr == "capture",
        )
        return instance

    @property
    def returncode(self) -> int | None:
        if self._process is not None:
            return self._process.returncode
        return self._status.returncode if self._status is not None else None

    @property
    def is_durable(self) -> bool:
        """Whether this handle can reconnect independently of its SSH channel."""

        return self._process is None

    async def poll(self) -> int | None:
        if self._process is not None:
            return self._process.returncode
        await self._refresh_status(
            deadline=time.monotonic() + PROCESS_POLL_RETRY_SECONDS
        )
        return self.returncode

    async def wait(self, *, timeout: float | None = None) -> int:
        resolved_timeout = self._timeout if timeout is None else timeout
        if self._process is None:
            return await asyncio.wait_for(
                self._wait_durable(), timeout=resolved_timeout
            )
        self.stdout.start()
        self.stderr.start()

        async def wait_and_drain() -> None:
            await self._process.wait_closed()
            await asyncio.gather(self.stdout.wait_eof(), self.stderr.wait_eof())

        try:
            await asyncio.wait_for(wait_and_drain(), timeout=resolved_timeout)
        except (OSError, asyncssh.Error) as exc:
            raise _attached_connection_error(exc) from exc
        returncode = self._process.returncode
        if returncode is None:
            raise ConnectionError(
                "sandbox SSH process ended without reporting an exit status"
            )
        return returncode

    async def _refresh_status(self, *, deadline: float | None) -> JobStatus:
        async with self._status_lock:
            current = self._status
            if current is not None and current.state.terminal:
                return current
            assert self._status_reader is not None
            following = await self._status_reader(deadline)
            self._record_status(current, following)
            return following

    def _record_status(
        self, current: JobStatus | None, following: JobStatus
    ) -> None:
        if (
            current is not None
            and following.state != current.state
            and not can_transition(current.state, following.state)
        ):
            raise SandboxFailedError(
                f"durable sandbox job {self.id} moved backward from "
                f"{current.state.value} to {following.state.value}"
            )
        self._status = following

    async def _accept_status(self, following: JobStatus) -> JobStatus:
        async with self._status_lock:
            current = self._status
            if current is not None and current.state.terminal:
                return current
            self._record_status(current, following)
            return following

    async def _wait_durable(self) -> int:
        delay = PROCESS_STATUS_INITIAL_DELAY_SECONDS
        while True:
            status = await self._refresh_status(deadline=None)
            if status.state.terminal:
                assert status.returncode is not None
                await self._maybe_auto_cleanup()
                return status.returncode
            await asyncio.sleep(delay)
            delay = min(PROCESS_STATUS_MAX_DELAY_SECONDS, delay * 2.0)

    async def _is_terminal(self) -> bool:
        return (await self._refresh_status(deadline=None)).state.terminal

    async def _stream_reached_eof(self, _stream: str) -> None:
        await self._maybe_auto_cleanup()

    async def _maybe_auto_cleanup(self) -> None:
        if self._process is not None or self._retain or self._cleaned:
            return
        status = self._status
        stdout = cast(_DurableReader[T], self.stdout)
        stderr = cast(_DurableReader[T], self.stderr)
        if status is not None and status.state.terminal and stdout.eof and stderr.eof:
            task = self._cleanup_task
            if task is None or task.done():
                task = asyncio.create_task(self._cleanup_remote())
                task.add_done_callback(_consume_cleanup_result)
                self._cleanup_task = task

    async def _cleanup_remote(self) -> None:
        async with self._cleanup_lock:
            if self._cleaned:
                return
            assert self._cleanup_job is not None
            await self._cleanup_job()
            self._cleaned = True

    async def cleanup(self) -> None:
        """Wait for a durable job, then remove all of its remote artifacts."""

        if self._process is not None:
            raise io.UnsupportedOperation("attached commands have no durable artifacts")
        await self._wait_durable()
        await self._cleanup_remote()

    async def terminate(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except (OSError, asyncssh.Error) as exc:
                raise _attached_connection_error(exc) from exc
            return
        async with self._termination_lock:
            status = self._status
            if status is None:
                status = await self._refresh_status(deadline=None)
            if status.state.terminal:
                return
            while status.pid is None:
                await asyncio.sleep(PROCESS_STATUS_INITIAL_DELAY_SECONDS)
                status = await self._refresh_status(deadline=None)
                if status.state.terminal:
                    return
            assert self._signal_job is not None
            status = await self._accept_status(await self._signal_job("TERM", status.pid))
            if status.state.terminal:
                await self._maybe_auto_cleanup()
                return

            deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
            delay = PROCESS_STATUS_INITIAL_DELAY_SECONDS
            while time.monotonic() < deadline:
                try:
                    status = await self._refresh_status(deadline=deadline)
                except ConnectionError:
                    break
                if status.state.terminal:
                    await self._maybe_auto_cleanup()
                    return
                await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(PROCESS_STATUS_MAX_DELAY_SECONDS, delay * 2.0)

            status = await self._accept_status(await self._signal_job("KILL", status.pid))
            if not status.state.terminal:
                raise SandboxFailedError(
                    f"durable sandbox job {self.id} did not stop after SIGKILL"
                )
            await self._maybe_auto_cleanup()

    def __del__(self) -> None:
        # Finalizers cannot make asynchronous cleanup reliable. When an event
        # loop is still active, schedule the same idempotent cleanup path; the
        # explicit cleanup method remains the deterministic API.
        if (
            getattr(self, "_process", None) is not None
            or getattr(self, "_retain", True)
            or getattr(self, "_cleaned", True)
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.cleanup())
        task.add_done_callback(_consume_cleanup_result)


def _consume_cleanup_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


def _attached_connection_error(error: BaseException) -> ConnectionError:
    return ConnectionError(
        "attached SSH process connection was lost; its remote state is unknown. "
        "Use pty=False for reconnectable unattended commands"
        + (f": {error}" if str(error) else "")
    )


__all__ = ["Process"]
