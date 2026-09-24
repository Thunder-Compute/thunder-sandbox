"""An owned SSH connection tunneling one sandbox service to loopback."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import TracebackType

import asyncssh

from .._common.lifecycle import finish_cleanup


class Tunnel:
    def __init__(
        self,
        connection: asyncssh.SSHClientConnection,
        listener: asyncssh.SSHListener,
        on_close: Callable[[Tunnel], None],
    ) -> None:
        self._connection = connection
        self._listener = listener
        self._on_close = on_close
        self.port = listener.get_port()
        self.host = "127.0.0.1"
        self._closed = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def closed(self) -> bool:
        return self._closed or self._connection.is_closed()

    async def close(self) -> None:
        self._closed = True
        self._listener.close()
        # A dedicated connection also closes accepted streams, without touching jobs.
        self._connection.close()
        try:
            await asyncio.wait_for(self._connection.wait_closed(), timeout=5)
        finally:
            self._on_close(self)

    async def __aenter__(self) -> Tunnel:
        if self.closed:
            raise RuntimeError("tunnel is closed")
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        await finish_cleanup(self.close())
