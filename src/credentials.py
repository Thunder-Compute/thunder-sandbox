"""The SSH credential a client presents to its organization's sandboxes.

One key is generated on this machine and never leaves it. The API signs its
public half into a certificate that every sandbox in the organization accepts,
so a machine that created no sandbox can still reach them all.

The credential is cached on disk so it survives across processes, but that is
only ever best effort: a read-only home, a missing HOME, or a container with no
writable mount must not stop a client from connecting. Whatever cannot be
persisted is kept in memory for the life of the process instead. Either way the
material ends up in a file, because ssh reads its key and certificate from
paths, not from this process.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import SandboxError

if TYPE_CHECKING:
    from .client import Client

# Renew this far before expiry so a certificate cannot lapse midway through a
# long command that was valid when it started.
RENEWAL_MARGIN_SECONDS = 15 * 60

# The credential in use by this process. It is the authority while the process
# lives; the on-disk copy is a cache that may be absent or unwritable.
_current: "_Credential | None" = None
_scratch: Path | None = None


class _Credential:
    """A usable key and certificate, wherever they happen to live."""

    def __init__(self, key_path: Path, certificate_path: Path, persisted: bool) -> None:
        self.key_path = key_path
        self.certificate_path = certificate_path
        self.persisted = persisted

    def is_usable(self) -> bool:
        return (
            self.key_path.exists()
            and self.certificate_path.exists()
            and _certificate_is_usable(self.certificate_path)
        )


def ensure_credential(client: "Client") -> tuple[Path, Path]:
    """Return (private key, certificate), minting or renewing as needed.

    Reuses the process credential, then the cached one on disk, and only asks
    the API when neither is usable.
    """
    global _current

    if _current is not None and _current.is_usable():
        return _current.key_path, _current.certificate_path

    adopted = _load_persisted(client)
    if adopted is not None:
        _current = adopted
        return adopted.key_path, adopted.certificate_path

    _current = _mint(client)
    return _current.key_path, _current.certificate_path


def _load_persisted(client: "Client") -> _Credential | None:
    """Adopt the cached credential when it is still good."""
    paths = client.config.paths
    candidate = _Credential(paths.ssh_key, paths.ssh_certificate, persisted=True)
    return candidate if candidate.is_usable() else None


def _mint(client: "Client") -> _Credential:
    """Obtain a fresh certificate, preferring the cache but never requiring it."""
    paths = client.config.paths
    key_path, persisted = _ensure_key(paths.sandbox_keys, paths.ssh_key)

    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
    response = client._request("POST", "/sandboxes/ssh-certificate", {"ssh_public_key": public_key})
    line = str(response.get("ssh_certificate", "")).strip()
    if not line:
        raise SandboxError("Thunder did not return an SSH certificate")

    certificate_path = paths.ssh_certificate if persisted else _scratch_dir() / "id_ed25519-cert.pub"
    if not _write(certificate_path, line + "\n", 0o644):
        # The key is on disk but the certificate is not; keep the pair together
        # in the scratch directory so both are readable by ssh.
        certificate_path = _scratch_dir() / "id_ed25519-cert.pub"
        if not _write(certificate_path, line + "\n", 0o644):
            raise SandboxError(f"could not write an SSH certificate to {certificate_path}")
        persisted = False
    return _Credential(key_path, certificate_path, persisted=persisted)


def _ensure_key(directory: Path, key_path: Path) -> tuple[Path, bool]:
    """Return a usable private key path and whether it is the persistent one.

    An existing cached key is reused so certificates already issued for it stay
    valid. If the cache cannot hold a key, one is generated in scratch space for
    this process instead.
    """
    if key_path.exists() and key_path.with_suffix(".pub").exists():
        return key_path, True
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _generate_key(key_path)
        return key_path, True
    except (OSError, SandboxError):
        # No writable cache. A per-process key still authenticates: the API
        # signs whatever public half it is given.
        scratch_key = _scratch_dir() / "id_ed25519"
        if not scratch_key.exists():
            _generate_key(scratch_key)
        return scratch_key, False


def _generate_key(key_path: Path) -> None:
    key_path.unlink(missing_ok=True)
    key_path.with_suffix(".pub").unlink(missing_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "thunder-sandbox", "-f", str(key_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not key_path.exists():
        raise SandboxError(f"could not generate an SSH key: {result.stderr.strip()}")
    key_path.chmod(0o600)


def _write(path: Path, content: str, mode: int) -> bool:
    """Write best effort. Returns False when the location cannot hold the file."""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(mode)
        return True
    except OSError:
        return False


def _scratch_dir() -> Path:
    """A writable directory for material that could not be cached.

    It is removed when the process exits, so a credential that never reached
    the cache does not outlive the process that minted it.
    """
    global _scratch
    if _scratch is None:
        _scratch = Path(tempfile.mkdtemp(prefix="thunder-sandbox-"))
        os.chmod(_scratch, 0o700)
        atexit.register(shutil.rmtree, _scratch, True)
    return _scratch


def _certificate_is_usable(certificate: Path) -> bool:
    """A certificate is usable while it has more than the renewal margin left.

    The expiry is read back from the certificate itself rather than remembered
    from the response, so a stale file left by an older client, a clock change,
    or a half-written renewal is detected rather than trusted.
    """
    if not certificate.exists():
        return False
    result = subprocess.run(
        ["ssh-keygen", "-L", "-f", str(certificate)], capture_output=True, text=True
    )
    if result.returncode != 0:
        return False
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line.startswith("Valid:"):
            continue
        # "Valid: from 2026-08-28T00:00:00 to 2026-08-28T12:00:00"
        parts = line.split(" to ")
        if len(parts) != 2:
            return False
        try:
            expiry = time.mktime(time.strptime(parts[1].strip(), "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return False
        return expiry - time.time() > RENEWAL_MARGIN_SECONDS
    return False
