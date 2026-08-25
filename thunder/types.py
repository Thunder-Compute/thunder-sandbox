"""Value types exposed by the Thunder sandbox client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class SandboxStatus(str, Enum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class GPU:
    type: str
    count: int = 1


@dataclass(frozen=True)
class Resources:
    cpu: int
    memory: int
    storage: int
    gpu: GPU | None = None


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

    @property
    def command(self) -> tuple[str, ...]:
        command = [
            "ssh", "-i", str(self.private_key_path), "-p", str(self.port),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
        ]
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
