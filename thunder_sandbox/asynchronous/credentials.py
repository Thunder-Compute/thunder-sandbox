"""The SSH credential a client presents to its organization's sandboxes.

One key is generated on this machine and never leaves it. Thunder signs its
public half into a certificate that every sandbox in the organization accepts,
so a machine that created no sandbox can still reach them all.

The credential is cached under TNR_HOME so it survives across processes, but
only ever as an optimisation: a read-only home or a container with no writable
mount must not stop a client from connecting. Nothing here is ever written to
disk in order to be used, because AsyncSSH takes keys and certificates as
values rather than paths.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import asyncssh

from .._common.config import ThunderPaths
from .._common.exceptions import SandboxError

if TYPE_CHECKING:
    from .client import Client

# Renew this far ahead of expiry so a certificate cannot lapse midway through
# a command that was valid when it started, and so a modest clock difference
# between this machine and Thunder cannot hand out a dead credential.
RENEWAL_MARGIN_SECONDS = 15 * 60


_SAME_BURST_SECONDS = 60.0


class SSHCredential:
    """A key and the certificate that authorises it, held in memory."""

    def __init__(
        self,
        key: asyncssh.SSHKey,
        certificate: asyncssh.SSHCertificate,
        expires_at: float,
    ) -> None:
        self.key = key
        self.certificate = certificate
        self.expires_at = expires_at

    def is_usable(self, *, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return self.expires_at - current > RENEWAL_MARGIN_SECONDS


class CredentialStore:
    """Resolves the organization credential, minting or renewing as needed."""

    def __init__(self, paths: ThunderPaths) -> None:
        self._paths = paths
        self._current: SSHCredential | None = None

    async def ensure(self, client: "Client") -> SSHCredential:
        """Reuse the process credential, then the cached one, then mint."""
        if self._current is not None and self._current.is_usable():
            return self._current
        cached = self._load()
        if cached is not None and cached.is_usable():
            self._current = cached
            return cached
        self._current = await self._mint(client, reuse=cached)
        return self._current

    def _load(self) -> SSHCredential | None:
        """Adopt the cached credential. Any damage means mint a fresh one."""
        try:
            key = asyncssh.import_private_key(
                self._paths.ssh_key.read_bytes()
            )
        except (OSError, asyncssh.Error, ValueError):
            return None
        try:
            certificate = asyncssh.import_certificate(
                self._paths.ssh_certificate.read_bytes()
            )
            meta = json.loads(self._paths.ssh_certificate_meta.read_text(encoding="utf-8"))
            expires_at = float(meta["expires_at"])
        except (OSError, asyncssh.Error, ValueError, KeyError, TypeError):
            # The key is still good; only the certificate needs reissuing.
            return SSHCredential(key, _NO_CERTIFICATE, 0.0)
        return SSHCredential(key, certificate, expires_at)

    async def _mint(
        self, client: "Client", *, reuse: SSHCredential | None
    ) -> SSHCredential:
        # Reuse the cached key so this machine keeps one identity; only the
        # certificate authorising it is short lived.
        key = reuse.key if reuse is not None else asyncssh.generate_private_key(
            "ssh-ed25519", comment="thunder-sandbox"
        )
        public_key = key.export_public_key().decode("utf-8").strip()
        response = await client._request(
            "POST", "/sandboxes/ssh-certificate", {"ssh_public_key": public_key}
        )
        line = str(response.get("ssh_certificate", "")).strip()
        if not line:
            raise SandboxError("Thunder did not return an SSH certificate")
        try:
            certificate = asyncssh.import_certificate(line)
        except (asyncssh.Error, ValueError) as exc:
            raise SandboxError(f"Thunder returned an unusable SSH certificate: {exc}") from exc
        expires_at = _expiry(response)
        self._save(key, line, expires_at, replace_key=reuse is None)
        return SSHCredential(key, certificate, expires_at)

    def _save(
        self,
        key: asyncssh.SSHKey,
        certificate: str,
        expires_at: float,
        *,
        replace_key: bool,
    ) -> None:
        """Cache best effort. A client that cannot write still connects.

        Opening several sandboxes at once has each one mint its own
        credential. They are all valid, so the first to arrive is written and
        the rest are simply used from memory: no caller waits on another's
        write, and the cache is not rewritten once per sandbox. A genuine
        renewal, hours later, carries a later expiry and does replace it.
        """
        # Skip only when the cache already holds a credential that is itself
        # usable and just as fresh, which is exactly the case where a sibling
        # from the same burst got here first. Anything else -- an empty cache, a
        # corrupt key, a certificate due for renewal -- is written. Asking the
        # cache what it actually holds keeps this independent of whether some
        # earlier write landed.
        persisted = self._load()
        if (
            persisted is not None
            and persisted.is_usable()
            and expires_at < persisted.expires_at + _SAME_BURST_SECONDS
        ):
            return
        try:
            self._paths.sandbox_keys.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Written whenever this key is new, which includes the case where a
            # damaged file is what forced a fresh one. Skipping on mere
            # existence would leave the unreadable key in place for every later
            # process to trip over.
            if replace_key:
                self._paths.ssh_key.write_bytes(key.export_private_key())
                self._paths.ssh_key.chmod(0o600)
            self._paths.ssh_certificate.write_text(certificate + "\n", encoding="utf-8")
            # The expiry is recorded beside the certificate rather than read
            # back out of it: AsyncSSH does not expose the validity window as
            # public API, and depending on its internals would break silently.
            self._paths.ssh_certificate_meta.write_text(
                json.dumps({"expires_at": expires_at}), encoding="utf-8"
            )
        except OSError:
            pass


def _expiry(response: dict[str, object]) -> float:
    raw = response.get("expires_at")
    if isinstance(raw, str) and raw:
        from datetime import datetime

        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    # Without a usable expiry the credential is treated as already due for
    # renewal rather than trusted indefinitely.
    return 0.0


class _NoCertificate:
    """Placeholder for a key whose certificate is missing or unreadable."""


_NO_CERTIFICATE = _NoCertificate()


__all__ = ["CredentialStore", "SSHCredential", "RENEWAL_MARGIN_SECONDS"]
