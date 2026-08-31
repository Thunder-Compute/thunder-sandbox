"""Authenticated HTTP client for Thunder sandboxes."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

from .config import ClientConfig
from .exceptions import AuthenticationError, ConflictError, ConnectionError, InvalidRequestError, NotFoundError, RetryableError, ThunderError

if TYPE_CHECKING:
    from .sandbox import Sandbox

USER_AGENT = "thunder-python-sdk/0.1.0"


class Client:
    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig.from_cli()
        if not self.config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        self._closed = False

    @classmethod
    def from_cli(cls) -> "Client":
        return cls(ClientConfig.from_cli())

    def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
        *,
        timeout: float | None = 30,
    ) -> dict[str, Any]:
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
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                parsed = json.loads(payload)
                message = parsed.get("message") or f"Thunder API returned HTTP {exc.code}"
            except (json.JSONDecodeError, AttributeError):
                message = f"Thunder API returned HTTP {exc.code}"
            error_type = {400: InvalidRequestError, 401: AuthenticationError, 403: AuthenticationError, 404: NotFoundError, 408: RetryableError, 409: ConflictError}.get(exc.code, ThunderError)
            raise error_type(message) from exc
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
        return self.get_sandbox(name)

    def list_sandboxes(self) -> Iterator["Sandbox"]:
        from .sandbox import Sandbox
        page_token = ""
        while True:
            response = self._request("GET", "/sandboxes", query={"limit": 100, "page_token": page_token})
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
