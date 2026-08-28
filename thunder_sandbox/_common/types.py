"""Value types exposed by the Thunder sandbox client."""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import InvalidRequestError
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
    # Where the organization credential is cached, for callers that shell out
    # to ssh. The SDK's own connections do not read these: it holds the key and
    # certificate as values, so a client whose cache is unwritable still works.
    private_key_path: Path | None = None
    certificate_path: Path | None = None
    known_hosts_path: Path | None = None

    @property
    def command(self) -> tuple[str, ...]:
        if self.private_key_path is None or self.certificate_path is None:
            raise InvalidRequestError(
                "no SSH credential is cached on this machine; connect through "
                "the SDK, which does not require one on disk"
            )
        command = [
            "ssh", "-i", str(self.private_key_path),
            "-o", f"CertificateFile={self.certificate_path}",
            "-p", str(self.port),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
        ]
        command.extend(("-o", "StrictHostKeyChecking=accept-new"))
        if self.known_hosts_path is not None:
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
