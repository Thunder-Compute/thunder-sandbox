"""Configuration shared with the Thunder CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .exceptions import AuthenticationError, InvalidRequestError

DEFAULT_API_URL = "https://api.thundercompute.com:8443"


class ThunderPaths:
    """Locations of credentials and sandbox SSH material."""

    def __init__(self, root: Path | None = None) -> None:
        configured_root = os.environ.get("TNR_HOME")
        self.root = Path(root or configured_root or Path.home() / ".thunder").expanduser()
        self.credentials = self.root / "cli_config.json"
        self.sandbox_keys = self.root / "sandbox_keys"
        self.known_hosts = self.root / "known_hosts"

    def sandbox_private_key(self, name: str) -> Path:
        _validate_sandbox_name(name)
        return self.sandbox_keys / name

    def sandbox_public_key(self, name: str) -> Path:
        return self.sandbox_private_key(name).with_suffix(".pub")

    # One key and one certificate serve every sandbox in the organization, so
    # they are not named after any sandbox. The key is generated once on this
    # machine and reused; only the certificate is refreshed.
    @property
    def ssh_key(self) -> Path:
        return self.sandbox_keys / "id_ed25519"

    @property
    def ssh_certificate(self) -> Path:
        return self.sandbox_keys / "id_ed25519-cert.pub"


class ClientConfig:
    """Resolved credentials. Explicit values, env, then CLI state win."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_token: str | None = None,
        paths: ThunderPaths | None = None,
    ) -> None:
        self.paths = paths or ThunderPaths()
        cli = _read_json(self.paths.credentials)
        project = _read_json(Path.cwd() / ".thunder.json")
        self.api_token = api_token or os.environ.get("TNR_API_TOKEN") or cli.get("token")
        self.api_url = str(
            api_url
            or os.environ.get("TNR_API_URL")
            or project.get("api_url")
            or cli.get("api_url")
            or DEFAULT_API_URL
        ).rstrip("/")

    @classmethod
    def from_cli(cls, *, root: Path | None = None) -> "ClientConfig":
        config = cls(paths=ThunderPaths(root))
        if not config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        return config


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidRequestError(f"could not read Thunder configuration at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidRequestError(f"Thunder configuration at {path} must be a JSON object")
    return value


def _validate_sandbox_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise InvalidRequestError(f"invalid sandbox name: {name!r}")
