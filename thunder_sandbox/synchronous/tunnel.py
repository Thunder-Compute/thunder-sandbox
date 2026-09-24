"""Blocking and awaitable access to an owned SSH tunnel."""

from __future__ import annotations

from types import TracebackType

from .._common.lifecycle import finish_cleanup
from ..asynchronous.tunnel import Tunnel as NativeTunnel
from ._bridge import AsyncBridge


class Tunnel:
    def __init__(self, bridge: AsyncBridge, tunnel: NativeTunnel) -> None:
        self._bridge = bridge
        self._tunnel = tunnel

    @property
    def host(self) -> str:
        return self._tunnel.host

    @property
    def port(self) -> int:
        return self._tunnel.port

    @property
    def address(self) -> str:
        return self._tunnel.address

    @property
    def closed(self) -> bool:
        return self._tunnel.closed

    def close(self) -> None:
        if not self.closed:
            self._bridge.run(self._tunnel.close())

    async def close_async(self) -> None:
        if not self.closed:
            await finish_cleanup(self._bridge.run_async(self._tunnel.close()))

    def __enter__(self) -> Tunnel:
        if self.closed:
            raise RuntimeError("tunnel is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Tunnel:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        await self.close_async()
