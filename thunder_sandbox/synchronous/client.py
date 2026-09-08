"""Blocking adapter over the native asynchronous Thunder client."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._common.config import ClientConfig
from .._common.exceptions import ConnectionError
from .._common.types import GPUType, SandboxStatus
from ..asynchronous.client import Client as NativeClient
from ._bridge import AsyncBridge

if TYPE_CHECKING:
    from ..asynchronous.sandbox import Sandbox as NativeSandbox
    from ..image import Image, ResolvedImage
    from .sandbox import Sandbox


class Client:
    def __init__(self, config: ClientConfig | None = None) -> None:
        self._client = NativeClient(config)
        self._bridge = AsyncBridge()
        self.config = self._client.config
        self._closed = False
        self._sandboxes: set[NativeSandbox] = set()

    @classmethod
    def from_cli(cls) -> "Client":
        return cls(ClientConfig.from_cli())

    def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ConnectionError("client is closed")
        return self._bridge.run(self._client._request(method, path, body, query))

    async def _request_async(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ConnectionError("client is closed")
        return await self._bridge.run_async(
            self._client._request(method, path, body, query)
        )

    def create_sandbox(
        self,
        *args: str,
        name: str | None = None,
        env: Mapping[str, str | None] | None = None,
        timeout: int | None = 300,
        cpu: int | None = None,
        memory: int | None = None,
        storage: int | None = None,
        gpu_type: GPUType | None = None,
        gpu_count: int | None = None,
        image: "Image | None" = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return Sandbox.create(
            *args,
            name=name,
            env=env,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            client=self,
        )

    def resolve_image(
        self, image: "Image", *, timeout: float | None = 7200
    ) -> "ResolvedImage":
        if self._closed:
            raise ConnectionError("client is closed")
        return self._bridge.run(self._client.resolve_image(image, timeout=timeout))

    async def resolve_image_async(
        self, image: "Image", *, timeout: float | None = 7200
    ) -> "ResolvedImage":
        if self._closed:
            raise ConnectionError("client is closed")
        return await self._bridge.run_async(
            self._client.resolve_image(image, timeout=timeout)
        )

    async def create_sandbox_async(
        self,
        *args: str,
        name: str | None = None,
        env: Mapping[str, str | None] | None = None,
        timeout: int | None = 300,
        cpu: int | None = None,
        memory: int | None = None,
        storage: int | None = None,
        gpu_type: GPUType | None = None,
        gpu_count: int | None = None,
        image: "Image | None" = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.create_async(
            *args,
            name=name,
            env=env,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            client=self,
        )

    def _wrap_sandbox(self, sandbox: "NativeSandbox") -> "Sandbox":
        from .sandbox import Sandbox

        self._sandboxes.add(sandbox)
        return Sandbox(self, sandbox)

    def get_sandbox(self, sandbox_id: str) -> "Sandbox":

        sandbox = self._bridge.run(
            self._client.get_sandbox(sandbox_id)
        )
        return self._wrap_sandbox(sandbox)

    async def get_sandbox_async(self, sandbox_id: str) -> "Sandbox":
        sandbox = await self._bridge.run_async(self._client.get_sandbox(sandbox_id))
        return self._wrap_sandbox(sandbox)

    def get_sandbox_by_name(self, name: str) -> "Sandbox":

        sandbox = self._bridge.run(
            self._client.get_sandbox_by_name(name)
        )
        return self._wrap_sandbox(sandbox)

    async def get_sandbox_by_name_async(self, name: str) -> "Sandbox":
        sandbox = await self._bridge.run_async(
            self._client.get_sandbox_by_name(name)
        )
        return self._wrap_sandbox(sandbox)

    def list_sandboxes(
        self, *, status: str | SandboxStatus = "active"
    ) -> Iterator["Sandbox"]:

        iterator = self._client.list_sandboxes(status=status)
        try:
            while True:
                try:
                    sandbox = self._bridge.run(anext(iterator))
                except StopAsyncIteration:
                    return
                yield self._wrap_sandbox(sandbox)
        finally:
            self._bridge.run(iterator.aclose())

    async def list_sandboxes_async(
        self, *, status: str | SandboxStatus = "active"
    ) -> AsyncGenerator["Sandbox", None]:
        iterator = self._client.list_sandboxes(status=status)
        try:
            while True:
                try:
                    sandbox = await self._bridge.run_async(anext(iterator))
                except StopAsyncIteration:
                    return
                yield self._wrap_sandbox(sandbox)
        finally:
            await self._bridge.run_async(iterator.aclose())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def close_resources() -> None:
            for sandbox in self._sandboxes:
                await sandbox._close_connection()
            self._sandboxes.clear()
            await self._client.close()

        try:
            self._bridge.run(close_resources())
        finally:
            self._bridge.close()

    async def close_async(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def close_resources() -> None:
            for sandbox in self._sandboxes:
                await sandbox._close_connection()
            self._sandboxes.clear()
            await self._client.close()

        try:
            await self._bridge.run_async(close_resources())
        finally:
            self._bridge.close()

    def __enter__(self) -> "Client":
        if self._closed:
            raise ConnectionError("client is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    async def __aenter__(self) -> "Client":
        if self._closed:
            raise ConnectionError("client is closed")
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        await self.close_async()


__all__ = ["Client"]
