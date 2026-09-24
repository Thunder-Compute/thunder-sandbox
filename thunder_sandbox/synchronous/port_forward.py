"""Blocking and awaitable access to an owned port forward."""

from __future__ import annotations

from types import TracebackType

from .._common.lifecycle import finish_cleanup
from ..asynchronous.port_forward import PortForward as NativePortForward
from ._bridge import AsyncBridge


class PortForward:
    def __init__(self, bridge: AsyncBridge, forward: NativePortForward) -> None:
        self._bridge = bridge
        self._forward = forward

    @property
    def host(self) -> str:
        return self._forward.host

    @property
    def port(self) -> int:
        return self._forward.port

    @property
    def address(self) -> str:
        return self._forward.address

    @property
    def closed(self) -> bool:
        return self._forward.closed

    def close(self) -> None:
        if not self.closed:
            self._bridge.run(self._forward.close())

    async def close_async(self) -> None:
        if not self.closed:
            await finish_cleanup(self._bridge.run_async(self._forward.close()))

    def __enter__(self) -> PortForward:
        if self.closed:
            raise RuntimeError("port forward is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> PortForward:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        await self.close_async()
