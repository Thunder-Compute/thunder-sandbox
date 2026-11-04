"""Value types exposed by the Thunder sandbox client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from .exceptions import InvalidRequestError

OutputMode = Literal["capture", "discard"]


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


class SnapshotStatus(str, Enum):
    CREATING = "creating"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "SnapshotStatus":
        # As for sandboxes: a newer API's status must not break this client.
        return cls.UNKNOWN

    @property
    def terminal(self) -> bool:
        """Whether capture has finished, successfully or not.

        ``DELETING`` is not terminal for capture: the snapshot is on its way
        out, and a read after deletion reports it as not found.
        """
        return self in (SnapshotStatus.READY, SnapshotStatus.FAILED)


class GPUType(str, Enum):
    A6000 = "A6000"
    A100 = "A100"       # NVIDIA A100 40 GiB
    A100XL = "A100XL"   # NVIDIA A100 80 GiB
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
class Pricing:
    """USD per resource-hour; memory_gb and storage_gb are per GiB."""

    vcpu: float
    storage_gb: float
    memory_gb: float
    gpu: dict[GPUType, float]


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
        # Sandboxes are short-lived and the node reuses forwarded ports, so a
        # known-hosts entry outlives the sandbox that created it and makes ssh
        # refuse the next one. Nothing verifies the first connection anyway, so
        # keep no file rather than one guaranteed to go stale.
        command.extend((
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ))
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
    image_id: str | None = None
    network_policy_generation: int | None = None
    desired_network_policy_generation: int | None = None
    # The snapshot this sandbox was restored from, if any.
    snapshot_id: str | None = None


@dataclass(frozen=True)
class SnapshotInfo:
    """A durable disk snapshot of a sandbox.

    ``storage_gib`` is the smallest disk a restore receives; the API grows a
    smaller requested disk to it. ``image_id`` is the source sandbox's image,
    which a restored sandbox inherits.
    """

    id: str
    name: str
    status: SnapshotStatus
    source_sandbox_id: str
    storage_gib: int
    created_at: datetime
    image_id: str | None = None
    ready_at: datetime | None = None
    expires_at: datetime | None = None
    failure_code: str | None = None
    failure: str | None = None
