"""Shared SSH connection ownership and retry policy."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeVar

import asyncssh

from .._common.exceptions import ConnectionError


SSH_RETRY_INITIAL_SECONDS = 0.25
SSH_RETRY_MAX_SECONDS = 5.0
SSH_RETRY_JITTER_RATIO = 0.2
SSH_CONNECT_TIMEOUT_SECONDS = 15.0

T = TypeVar("T")
OpenConnection = Callable[[], Awaitable[asyncssh.SSHClientConnection]]
SSHOperation = Callable[[asyncssh.SSHClientConnection], Awaitable[T]]


class RetryableSSHOperationError(OSError):
    """An ambiguous short SSH operation which is safe to reconcile and retry."""


def is_transient_ssh_error(error: BaseException) -> bool:
    """Return whether reconnecting can plausibly recover from ``error``."""

    # Retrying these either hides a trust failure or repeatedly presents a
    # credential which was already renewed and rejected.
    if isinstance(error, (asyncssh.HostKeyNotVerifiable, asyncssh.PermissionDenied)):
        return False
    return isinstance(error, (OSError, asyncssh.Error, RetryableSSHOperationError))


class SSHConnectionManager:
    """Own one reusable connection and retry explicitly idempotent operations."""

    def __init__(
        self,
        open_connection: OpenConnection,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._open_connection = open_connection
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._connection: asyncssh.SSHClientConnection | None = None
        self._connection_lock = asyncio.Lock()
        self._closed = False

    async def get(self) -> asyncssh.SSHClientConnection:
        """Return the healthy cached connection, opening at most one replacement."""

        if self._closed:
            raise ConnectionError("sandbox SSH connection manager is closed")
        connection = self._connection
        if connection is not None and not connection.is_closed():
            return connection
        async with self._connection_lock:
            if self._closed:
                raise ConnectionError("sandbox SSH connection manager is closed")
            connection = self._connection
            if connection is not None and not connection.is_closed():
                return connection
            connection = await self._open_connection()
            self._connection = connection
            return connection

    async def discard(self, connection: asyncssh.SSHClientConnection) -> None:
        """Close a failed connection without evicting a newer replacement."""

        if self._connection is connection:
            self._connection = None
        connection.close()
        with suppress(Exception):
            await connection.wait_closed()

    async def run(
        self,
        operation: SSHOperation[T],
        *,
        name: str,
        deadline: float | None = None,
    ) -> T:
        """Run an idempotent operation, reconnecting after transport failures.

        ``deadline`` is an absolute value in this manager's monotonic clock.
        Without one, retries continue until success, cancellation, a permanent
        SSH error, or manager shutdown.
        """

        delay = SSH_RETRY_INITIAL_SECONDS
        while True:
            connection: asyncssh.SSHClientConnection | None = None
            try:
                connection = await self.get()
                return await operation(connection)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                if not is_transient_ssh_error(exc):
                    raise
                if connection is not None:
                    await self.discard(connection)
                if deadline is not None:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise ConnectionError(
                            f"{name} did not complete before its SSH retry deadline"
                        ) from exc
                jitter = self._jitter(0.0, delay * SSH_RETRY_JITTER_RATIO)
                pause = delay + jitter
                if deadline is not None:
                    pause = min(pause, max(0.0, deadline - self._clock()))
                await self._sleep(pause)
                delay = min(SSH_RETRY_MAX_SECONDS, delay * 2.0)

    async def close(self) -> None:
        """Close only the transport; remote detached jobs remain untouched."""

        async with self._connection_lock:
            if self._closed:
                return
            self._closed = True
            connection = self._connection
            self._connection = None
        if connection is not None:
            connection.close()
            with suppress(Exception):
                await connection.wait_closed()


__all__ = [
    "RetryableSSHOperationError",
    "SSHConnectionManager",
    "SSH_CONNECT_TIMEOUT_SECONDS",
    "SSH_RETRY_INITIAL_SECONDS",
    "SSH_RETRY_JITTER_RATIO",
    "SSH_RETRY_MAX_SECONDS",
    "is_transient_ssh_error",
]
