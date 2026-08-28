"""Native asynchronous HTTP client for Thunder sandboxes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import aiohttp

from .._common.config import ClientConfig
from .._common.exceptions import (
    AuthenticationError,
    CapacityError,
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ThunderError,
)
from .._common.types import GPUType, SandboxStatus
from .._version import __version__

if TYPE_CHECKING:
    from .sandbox import Sandbox

USER_AGENT = f"thunder-python-sdk/{__version__}"

_ERROR_CODES: dict[str, type[ThunderError]] = {
    "sandbox_capacity_unavailable": CapacityError,
    "sandbox_scheduler_unavailable": ServiceUnavailableError,
    "sandbox_scheduler_timeout": ServiceUnavailableError,
    "sandbox_already_exists": ConflictError,
    "sandbox_name_in_use": ConflictError,
    "sandbox_scheduler_rejected_request": InvalidRequestError,
    "rate_limit_exceeded": RateLimitError,
}

_ERROR_STATUSES: dict[int, type[ThunderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
    429: RateLimitError,
    502: ServiceUnavailableError,
    503: ServiceUnavailableError,
    504: ServiceUnavailableError,
}


def _api_error(
    status: int, code: str | None, message: str, headers: object
) -> ThunderError:
    error_type = _ERROR_CODES.get(code or "") or _ERROR_STATUSES.get(status, ThunderError)
    retry_after = None
    getter = getattr(headers, "get", None)
    if getter is not None:
        try:
            retry_after = float(getter("Retry-After"))
        except (TypeError, ValueError):
            retry_after = None
    return error_type(message, code=code, status=status, retry_after=retry_after)


class Client:
    """An asynchronous Thunder API client backed by :mod:`aiohttp`."""

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig.from_cli()
        if not self.config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    @classmethod
    def from_cli(cls) -> "Client":
        return cls(ClientConfig.from_cli())

    def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise ConnectionError("client is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json",
                    "Thunder-Client": "PYTHON-SDK",
                    "User-Agent": USER_AGENT,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        session = self._get_session()
        url = f"{self.config.api_url}/v1{path}"
        params = (
            {key: str(value) for key, value in query.items() if value not in (None, "")}
            if query
            else None
        )
        try:
            async with session.request(method, url, json=body, params=params) as response:
                payload = await response.read()
                if response.status >= 400:
                    code = None
                    try:
                        parsed = json.loads(payload)
                        message = (
                            parsed.get("message")
                            or f"Thunder API returned HTTP {response.status}"
                        )
                        code = parsed.get("error")
                    except (json.JSONDecodeError, AttributeError):
                        message = f"Thunder API returned HTTP {response.status}"
                    raise _api_error(
                        response.status, code, message, response.headers
                    )
        except ThunderError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise ConnectionError(f"could not communicate with Thunder: {exc}") from exc

        if not payload:
            return {}
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Thunder returned an invalid JSON response") from exc
        if not isinstance(result, dict):
            raise ConnectionError("Thunder returned an unexpected response")
        return result

    async def create_sandbox(
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
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        ssh_public_key: str | None = None,
        ssh_private_key: str | None = None,
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.create(
            *args,
            name=name,
            env=env,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            ssh_public_key=ssh_public_key,
            ssh_private_key=ssh_private_key,
            client=self,
        )

    async def get_sandbox(self, sandbox_id: str) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_id(sandbox_id, client=self)

    async def get_sandbox_by_name(self, name: str) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_name(name, client=self)

    async def list_sandboxes(
        self, *, status: str | SandboxStatus = "active"
    ) -> AsyncGenerator["Sandbox", None]:
        from .sandbox import Sandbox

        wanted = status.value if isinstance(status, SandboxStatus) else str(status)
        page_token = ""
        while True:
            response = await self._request(
                "GET",
                "/sandboxes",
                query={"limit": 100, "status": wanted, "page_token": page_token},
            )
            for item in response.get("sandboxes", []):
                if isinstance(item, dict):
                    yield Sandbox._from_response(self, item)
            page_token = str(response.get("next_page_token", ""))
            if not page_token:
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "Client":
        if self._closed:
            raise ConnectionError("client is closed")
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        await self.close()
