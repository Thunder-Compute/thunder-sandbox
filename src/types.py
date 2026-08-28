"""Value types exposed by the Thunder sandbox client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class SandboxStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    FINISHED = "finished"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "SandboxStatus":
        # A status this client does not know is reported as UNKNOWN rather than
        # raising: a newer API must not break an older client mid-call.
        return cls.UNKNOWN

    @property
    def live(self) -> bool:
        """Whether the sandbox still exists and holds its name."""
        return self in (SandboxStatus.CREATED, SandboxStatus.READY)

    @property
    def terminal(self) -> bool:
        return self in (SandboxStatus.FINISHED, SandboxStatus.FAILED)


class GPUType(str, Enum):
    A6000 = "A6000"
    A100 = "A100"
    H100 = "H100"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "GPUType | None":
        if not isinstance(value, str):
            return None
        wanted = value.strip().upper()
        for member in cls:
            if member.value.upper() == wanted:
                return member
        return cls.UNKNOWN


@dataclass(frozen=True)
class Resources:
    cpu: int
    memory: int
    storage: int
    gpu_type: GPUType | None = None
    gpu_count: int = 0


@dataclass(frozen=True)
class NetworkPolicy:
    internet_access: str
    outbound_cidr_allowlist: tuple[str, ...] = ()
    outbound_domain_allowlist: tuple[str, ...] = ()

    @property
    def block_network(self) -> bool:
        return self.internet_access == "closed"


@dataclass(frozen=True)
class SSHConnection:
    host: str
    port: int
    user: str
    private_key_path: Path
    known_hosts_path: Path | None = None
    certificate_path: Path | None = None

    @property
    def command(self) -> tuple[str, ...]:
        command = [
            "ssh", "-i", str(self.private_key_path), "-p", str(self.port),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
        ]
        if self.certificate_path is not None:
            # The sandbox trusts the authority that signed this, not the key
            # itself, which is why the same credential opens every sandbox.
            command.extend(("-o", f"CertificateFile={self.certificate_path}"))
        if self.known_hosts_path is None:
            command.extend(("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"))
        else:
            command.extend(("-o", f"UserKnownHostsFile={self.known_hosts_path}"))
        command.append(f"{self.user}@{self.host}")
        return tuple(command)


@dataclass(frozen=True)
class SandboxInfo:
    id: str
    name: str
    status: SandboxStatus
    resources: Resources
    network_policy: NetworkPolicy
    created_at: datetime
    expires_at: datetime | None = None
    ssh: SSHConnection | None = None
    failure_code: str | None = None
    failure: str | None = None
