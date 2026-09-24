"""A loopback service tunnel owned by the sandbox SSH manager."""

from __future__ import annotations

import asyncio
from types import TracebackType

import asyncssh

from .._common.exceptions import ConnectionError, SandboxTimeoutError, UnsupportedFeatureError
from .._common.lifecycle import finish_cleanup
from .._common.types import validate_port
from ._ssh import SSHConnectionManager




class Tunnel:
    def __init__(
        self, parent: SSHConnectionManager, owner: SSHConnectionManager,
        connection: asyncssh.SSHClientConnection, listener: asyncssh.SSHListener,
    ) -> None:
        self._parent = parent
        self._owner = owner
        self._connection = connection
        self._listener = listener
        self.port = listener.get_port()
        self.host = "127.0.0.1"
        self._closed = False

    @classmethod
    async def _open(
        cls, parent: SSHConnectionManager, remote_port: int, *,
        local_port: int = 0, timeout: float = 30,
    ) -> Tunnel:
        validate_port(remote_port)
        validate_port(local_port, allow_zero=True)
        owner = parent.dedicated()

        async def connect() -> Tunnel:
            connection = await owner.get()
            _, writer = await connection.open_connection("127.0.0.1", remote_port)
            writer.close()
            await writer.wait_closed()
            listener = await connection.forward_local_port(
                "127.0.0.1", local_port, "127.0.0.1", remote_port,
            )
            if connection.is_closed():
                listener.close()
                await listener.wait_closed()
                raise ConnectionError("sandbox SSH connection closed while opening tunnel")
            return cls(parent, owner, connection, listener)

        try:
            try:
                return await asyncio.wait_for(connect(), timeout)
            except BaseException:
                await finish_cleanup(parent.release(owner))
                raise
        except asyncssh.ChannelOpenError as exc:
            if exc.code == asyncssh.OPEN_ADMINISTRATIVELY_PROHIBITED:
                raise UnsupportedFeatureError(
                    "sandbox SSH forwarding is disabled; the platform must grant "
                    "permit-port-forwarding and allow local TCP forwarding to loopback",
                    code="ssh_forwarding_disabled",
                ) from exc
            raise ConnectionError(f"cannot reach sandbox port {remote_port}: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"opening sandbox port {remote_port} exceeded {timeout} seconds"
            ) from exc
        except (OSError, asyncssh.Error) as exc:
            raise ConnectionError(f"could not open sandbox tunnel: {exc}") from exc

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def closed(self) -> bool:
        return self._closed or self._connection.is_closed()

    async def close(self) -> None:
        self._closed = True
        self._listener.close()
        await finish_cleanup(self._parent.release(self._owner))

    async def __aenter__(self) -> Tunnel:
        if self.closed:
            raise RuntimeError("tunnel is closed")
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        await self.close()
