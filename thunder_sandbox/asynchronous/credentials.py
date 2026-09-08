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
import asyncio
import base64
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import asyncssh
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, SSHCertificate, SSHCertificateType, load_ssh_public_identity,
)

from .._common.config import ThunderPaths
from .._common.exceptions import SandboxError
from .._common.types import SSHCertificateIdentity

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
        try:
            self.validate(now=current)
        except SandboxError:
            return False
        return self.expires_at - current > RENEWAL_MARGIN_SECONDS

    def validate(self, identity: SSHCertificateIdentity | None = None, *, now: float | None = None) -> None:
        try:
            cert = load_ssh_public_identity(self.certificate.export_certificate())
            if not isinstance(cert, SSHCertificate) or cert.type != SSHCertificateType.USER:
                raise ValueError("expected an OpenSSH user certificate")
            cert.verify_cert_signature()
            current = time.time() if now is None else now
            if not cert.valid_after <= current < cert.valid_before:
                raise ValueError("SSH certificate is expired or not yet valid")
            self.expires_at = min(self.expires_at, float(cert.valid_before))
            public = cert.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
            if public.split()[:2] != self.key.export_public_key().split()[:2]:
                raise ValueError("SSH certificate does not match the local private key")
            if identity is not None:
                ca = cert.signature_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
                fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(base64.b64decode(ca.split()[1])).digest()).decode().rstrip("=")
                if fingerprint != identity.ca_fingerprint:
                    raise ValueError("SSH certificate CA fingerprint does not match the sandbox's expected authority")
                if identity.principal.encode() not in cert.valid_principals:
                    raise ValueError("SSH certificate does not contain the sandbox's expected principal")
        except Exception as exc:
            raise SandboxError(f"Invalid SSH certificate: {exc}") from exc


class CredentialStore:
    """Resolves the organization credential, minting or renewing as needed."""

    def __init__(self, paths: ThunderPaths) -> None:
        self._paths = paths
        self._current: SSHCredential | None = None
        self._selected: dict[SSHCertificateIdentity, SSHCredential] = {}
        self._lock = asyncio.Lock()
        self._key: asyncssh.SSHKey | None = None

    def _private_key(self) -> asyncssh.SSHKey:
        if self._key is not None:
            return self._key
        key = asyncssh.generate_private_key("ssh-ed25519", comment="thunder-sandbox")
        try:
            self._paths.sandbox_keys.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self._paths.sandbox_keys) as temporary:
                temporary.write(key.export_private_key())
                temporary.flush()
                try:
                    # Publish a complete key without overwriting a concurrent creator.
                    os.link(temporary.name, self._paths.ssh_key)
                except FileExistsError:
                    key = asyncssh.import_private_key(self._paths.ssh_key.read_bytes())
        except OSError:
            pass
        except (asyncssh.Error, ValueError) as exc:
            raise SandboxError("cached SSH private key is corrupt") from exc
        self._key = key
        return key

    async def ensure(self, client: "Client", identity: SSHCertificateIdentity | None = None) -> SSHCredential:
        async with self._lock:
            return await self._ensure(client, identity)

    async def _ensure(self, client: "Client", identity: SSHCertificateIdentity | None) -> SSHCredential:
        """Reuse the process credential, then the cached one, then mint."""
        current = self._selected.get(identity) if identity is not None else self._current
        if current is not None and current.is_usable():
            current.validate(identity)
            return current
        cached = self._load(identity)
        if cached is not None and cached.is_usable():
            self._remember(cached, identity)
            return cached
        minted = await self._mint(client, reuse=cached or self._current, identity=identity)
        self._remember(minted, identity)
        return minted

    def _remember(self, credential: SSHCredential, identity: SSHCertificateIdentity | None) -> None:
        self._current = credential
        if identity is not None:
            self._selected[identity] = credential

    async def renew(self, client: "Client", identity: SSHCertificateIdentity | None = None) -> SSHCredential:
        """Mint a replacement for a credential the sandbox would not accept.

        A cached certificate can be unexpired and still useless: signed by
        another environment's authority, or by one that has since rotated.
        Expiry cannot detect either, so the only evidence is the sandbox
        refusing it.
        """
        async with self._lock:
            minted = await self._mint(client, reuse=self._load(identity) or self._current, replace=True, identity=identity)
            self._remember(minted, identity)
            return minted

    def _load(self, identity: SSHCertificateIdentity | None = None) -> SSHCredential | None:
        """Adopt the cached credential. Any damage means mint a fresh one."""
        try:
            key = asyncssh.import_private_key(
                self._paths.ssh_key.read_bytes()
            )
        except (OSError, asyncssh.Error, ValueError):
            return None
        try:
            certificate = asyncssh.import_certificate(
                self._paths.certificate_for(identity).read_bytes()
            )
            parsed = load_ssh_public_identity(certificate.export_certificate())
            if not isinstance(parsed, SSHCertificate):
                raise ValueError("not an OpenSSH certificate")
            expires_at = float(parsed.valid_before)
            credential = SSHCredential(key, certificate, expires_at)
            credential.validate(identity)
        except (OSError, asyncssh.Error, ValueError, KeyError, TypeError, SandboxError):
            # The key is still good; only the certificate needs reissuing.
            return SSHCredential(key, _NO_CERTIFICATE, 0.0)
        return SSHCredential(key, certificate, expires_at)

    async def _mint(
        self,
        client: "Client",
        *,
        reuse: SSHCredential | None,
        replace: bool = False,
        identity: SSHCertificateIdentity | None = None,
    ) -> SSHCredential:
        # Reuse the cached key so this machine keeps one identity; only the
        # certificate authorising it is short lived.
        key = reuse.key if reuse is not None else self._private_key()
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
        credential = SSHCredential(key, certificate, expires_at)
        credential.validate(identity)
        self._save(
            key, line, expires_at, replace=replace, identity=identity
        )
        return credential

    def _save(
        self,
        key: asyncssh.SSHKey,
        certificate: str,
        expires_at: float,
        *,
        replace: bool = False,
        identity: SSHCertificateIdentity | None = None,
    ) -> None:
        """Cache best effort. A client that cannot write still connects.

        Opening several sandboxes at once has each one mint its own
        credential. They are all valid, so the first to arrive is written and
        the rest are simply used from memory: no caller waits on another's
        write, and the cache is not rewritten once per sandbox. A genuine
        renewal, hours later, carries a later expiry and does replace it.
        A replacement for a certificate the sandbox refused is written even
        when its expiry is no later: the cached one is the one that failed.
        """
        # Skip only when the cache already holds a credential that is itself
        # usable and just as fresh, which is exactly the case where a sibling
        # from the same burst got here first. Anything else -- an empty cache, a
        # corrupt key, a certificate due for renewal, a refused certificate --
        # is written. Asking the cache what it actually holds keeps this
        # independent of whether some earlier write landed.
        persisted = self._load(identity)
        if (
            not replace
            and persisted is not None
            and persisted.is_usable()
            and expires_at < persisted.expires_at + _SAME_BURST_SECONDS
        ):
            return
        try:
            self._paths.sandbox_keys.mkdir(mode=0o700, parents=True, exist_ok=True)
            _atomic_write(self._paths.certificate_for(identity), certificate + "\n")
            if identity is None:
                _atomic_write(self._paths.ssh_certificate_meta,
                    json.dumps({"expires_at": expires_at})
                )
        except OSError:
            pass


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(text.encode())
            temporary.flush()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


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
