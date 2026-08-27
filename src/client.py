"""Authenticated HTTP client for Thunder sandboxes."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

from .config import ClientConfig
from .types import SandboxStatus
from .exceptions import (
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

if TYPE_CHECKING:
    from .sandbox import Sandbox

USER_AGENT = "thunder-python-sdk/0.1.0"


# Conditions a caller acts on differently, keyed by the API's machine-readable
# error identifier. Selecting on the identifier rather than the HTTP status
# keeps distinct conditions that share a status apart: a sandbox request can be
# refused for want of capacity or because the scheduler itself is down, and both
# are 503, but only the first is worth retrying unchanged.
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


def _api_error(status: int, code: str | None, message: str, headers: object) -> ThunderError:
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
    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig.from_cli()
        if not self.config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        self._closed = False

    @classmethod
    def from_cli(cls) -> "Client":
        return cls(ClientConfig.from_cli())

    def _request(self, method: str, path: str, body: object | None = None, query: dict[str, object] | None = None) -> dict[str, Any]:
        if self._closed:
            raise ConnectionError("client is closed")
        url = f"{self.config.api_url}/v1{path}"
        if query:
            values = {key: str(value) for key, value in query.items() if value not in (None, "")}
            if values:
                url += "?" + urllib.parse.urlencode(values)
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.config.api_token}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Thunder-Client", "PYTHON-SDK")
        request.add_header("User-Agent", USER_AGENT)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            code = None
            try:
                parsed = json.loads(payload)
                message = parsed.get("message") or f"Thunder API returned HTTP {exc.code}"
                code = parsed.get("error")
            except (json.JSONDecodeError, AttributeError):
                message = f"Thunder API returned HTTP {exc.code}"
            raise _api_error(exc.code, code, message, exc.headers) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
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

    def create_sandbox(self, *args: str, **options: object) -> "Sandbox":
        from .sandbox import Sandbox
        return Sandbox.create(*args, client=self, **options)

    def get_sandbox(self, sandbox_id: str) -> "Sandbox":
        from .sandbox import Sandbox
        return Sandbox.from_id(sandbox_id, client=self)

    def get_sandbox_by_name(self, name: str) -> "Sandbox":
        """Find the live sandbox holding this name.

        Names are labels rather than addresses, so this searches instead of
        fetching by key. Prefer get_sandbox with an ID.
        """
        from .sandbox import Sandbox
        return Sandbox.from_name(name, client=self)

    def list_sandboxes(self, *, status: str | SandboxStatus = "active") -> Iterator["Sandbox"]:
        """Iterate sandboxes, newest first.

        Defaults to the sandboxes that still exist. Pass "all" for the
        organization's history, or a single status to narrow further.
        """
        from .sandbox import Sandbox
        wanted = status.value if isinstance(status, SandboxStatus) else str(status)
        page_token = ""
        while True:
            response = self._request("GET", "/sandboxes",
                                     query={"limit": 100, "status": wanted, "page_token": page_token})
            for item in response.get("sandboxes", []):
                if isinstance(item, dict):
                    yield Sandbox._from_response(self, item)
            page_token = str(response.get("next_page_token", ""))
            if not page_token:
                return

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
