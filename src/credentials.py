"""The SSH credential a client presents to its organization's sandboxes.

One key is generated on this machine and never leaves it. The API signs its
public half into a certificate that every sandbox in the organization accepts,
so a machine that created no sandbox can still reach them all. The certificate
expires on its own; this module mints a new one when it is close to doing so.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import SandboxError

if TYPE_CHECKING:
    from .client import Client

# Refresh this far before expiry so a certificate cannot lapse midway through
# a long-running command that was valid when it started.
RENEWAL_MARGIN_SECONDS = 15 * 60


def ensure_credential(client: "Client") -> tuple[Path, Path]:
    """Return (private key, certificate), minting or renewing as needed."""
    paths = client.config.paths
    paths.sandbox_keys.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = paths.ssh_key
    certificate = paths.ssh_certificate
    if not key.exists():
        _generate_key(key)
        # A certificate for a key that no longer exists is unusable.
        certificate.unlink(missing_ok=True)
    if not _certificate_is_usable(certificate):
        _mint(client, key, certificate)
    return key, certificate


def _generate_key(key: Path) -> None:
    key.unlink(missing_ok=True)
    key.with_suffix(".pub").unlink(missing_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "thunder-sandbox", "-f", str(key)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not key.exists():
        raise SandboxError(f"could not generate an SSH key: {result.stderr.strip()}")
    key.chmod(0o600)


def _mint(client: "Client", key: Path, certificate: Path) -> None:
    public_key = key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    response = client._request("POST", "/sandboxes/ssh-certificate", {"ssh_public_key": public_key})
    line = str(response.get("ssh_certificate", "")).strip()
    if not line:
        raise SandboxError("Thunder did not return an SSH certificate")
    certificate.write_text(line + "\n", encoding="utf-8")
    certificate.chmod(0o644)


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
