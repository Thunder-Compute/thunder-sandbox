"""Versioned on-sandbox job protocol used by durable SSH execution.

This module deliberately contains no SSH or filesystem operations. It defines
the values which cross that boundary so the launcher, reconnecting process
handle, and tests agree before durable execution replaces attached channels.
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping


JOB_PROTOCOL_VERSION = 1
JOB_ROOT = PurePosixPath("/var/tmp/thunder-sandbox/jobs")

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
OutputMode = Literal["capture", "discard"]


class JobState(str, Enum):
    """States persisted by the remote durable-job wrapper."""

    PREPARED = "prepared"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINATED = "terminated"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.TERMINATED,
        }


_TRANSITIONS = {
    # Readers poll snapshots and can legitimately miss intermediate writes.
    JobState.PREPARED: frozenset(
        {
            JobState.STARTING,
            JobState.RUNNING,
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.TERMINATED,
        }
    ),
    JobState.STARTING: frozenset(
        {
            JobState.RUNNING,
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.TERMINATED,
        }
    ),
    JobState.RUNNING: frozenset(
        {JobState.SUCCEEDED, JobState.FAILED, JobState.TERMINATED}
    ),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.TERMINATED: frozenset(),
}


def new_job_id() -> str:
    """Return an unpredictable path-safe id before remote submission begins."""

    return uuid.uuid4().hex


def validate_job_id(job_id: str) -> str:
    """Reject identifiers which could escape or alias the remote job root."""

    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("job ID must be 32 lowercase hexadecimal characters")
    return job_id


def can_transition(previous: JobState, following: JobState) -> bool:
    """Return whether a persisted state may advance to another state."""

    return following in _TRANSITIONS[previous]


@dataclass(frozen=True)
class RemoteJobPaths:
    """All remote paths owned by one job.

    A caller may override ``root`` in tests, but production paths are fixed and
    never contain user-controlled path components other than a validated id.
    """

    job_id: str
    root: PurePosixPath = JOB_ROOT

    def __post_init__(self) -> None:
        validate_job_id(self.job_id)
        if not self.root.is_absolute():
            raise ValueError("remote job root must be absolute")

    @property
    def directory(self) -> PurePosixPath:
        return self.root / self.job_id

    @property
    def specification(self) -> PurePosixPath:
        return self.directory / "spec.json"

    @property
    def launcher(self) -> PurePosixPath:
        return self.directory / "launch.sh"

    @property
    def execution_claim(self) -> PurePosixPath:
        # A hard link atomically publishes a fully written owner-PID file.
        return self.directory / "execution.claim"

    @property
    def status(self) -> PurePosixPath:
        return self.directory / "status.json"

    @property
    def stdout(self) -> PurePosixPath:
        return self.directory / "stdout"

    @property
    def stderr(self) -> PurePosixPath:
        return self.directory / "stderr"

    @property
    def launcher_stderr(self) -> PurePosixPath:
        return self.directory / "launcher.stderr"

    @property
    def termination_request(self) -> PurePosixPath:
        return self.directory / "termination.request"


def _validate_text(value: str, label: str, *, empty: bool = True) -> None:
    if not empty and not value:
        raise ValueError(f"{label} cannot be empty")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain NUL")


@dataclass(frozen=True)
class JobSpec:
    """Immutable structured command staged before a job is launched."""

    job_id: str
    argv: tuple[str, ...]
    workdir: str | None = None
    env: Mapping[str, str | None] | None = None
    container: str | None = None
    stdout: OutputMode = "capture"
    stderr: OutputMode = "capture"
    retain: bool = False

    def __post_init__(self) -> None:
        validate_job_id(self.job_id)
        if not self.argv:
            raise ValueError("job command cannot be empty")
        for argument in self.argv:
            _validate_text(argument, "command argument")
        if self.workdir is not None:
            _validate_text(self.workdir, "working directory", empty=False)
        if self.container is not None:
            _validate_text(self.container, "container name", empty=False)
        if self.stdout not in {"capture", "discard"}:
            raise ValueError("stdout must be 'capture' or 'discard'")
        if self.stderr not in {"capture", "discard"}:
            raise ValueError("stderr must be 'capture' or 'discard'")
        if not isinstance(self.retain, bool):
            raise ValueError("retain must be a boolean")
        for name, value in (self.env or {}).items():
            if not name or "=" in name or "\x00" in name:
                raise ValueError(f"invalid environment variable name: {name!r}")
            if value is not None:
                _validate_text(value, f"environment variable {name}")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env or {})))

    def to_json(self) -> str:
        return json.dumps(
            {
                "protocol": JOB_PROTOCOL_VERSION,
                "job_id": self.job_id,
                "argv": list(self.argv),
                "workdir": self.workdir,
                "env": dict(self.env or {}),
                "container": self.container,
                "stdout": self.stdout,
                "stderr": self.stderr,
                "retain": self.retain,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "JobSpec":
        """Decode a persisted spec, defaulting fields added within protocol v1."""

        try:
            raw = json.loads(value)
            if not isinstance(raw, dict):
                raise TypeError
            if raw.get("protocol") != JOB_PROTOCOL_VERSION:
                raise ValueError("unsupported durable-job protocol version")
            job_id = raw["job_id"]
            argv = raw["argv"]
            workdir = raw.get("workdir")
            env = raw.get("env")
            container = raw.get("container")
            stdout = raw.get("stdout", "capture")
            stderr = raw.get("stderr", "capture")
            retain = raw.get("retain", False)
            if not isinstance(job_id, str) or not isinstance(argv, list):
                raise TypeError
            if not all(isinstance(argument, str) for argument in argv):
                raise TypeError
            if workdir is not None and not isinstance(workdir, str):
                raise TypeError
            if env is not None and (
                not isinstance(env, dict)
                or not all(
                    isinstance(name, str)
                    and (item is None or isinstance(item, str))
                    for name, item in env.items()
                )
            ):
                raise TypeError
            if container is not None and not isinstance(container, str):
                raise TypeError
            if not isinstance(stdout, str) or not isinstance(stderr, str):
                raise TypeError
            if not isinstance(retain, bool):
                raise TypeError
            return cls(
                job_id,
                tuple(argv),
                workdir=workdir,
                env=env,
                container=container,
                stdout=stdout,  # type: ignore[arg-type]
                stderr=stderr,  # type: ignore[arg-type]
                retain=retain,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("invalid durable-job specification") from exc


@dataclass(frozen=True)
class JobStatus:
    """One atomically-written observation of a durable remote job."""

    state: JobState
    pid: int | None = None
    returncode: int | None = None

    def __post_init__(self) -> None:
        if self.pid is not None and self.pid <= 0:
            raise ValueError("job PID must be positive")
        if self.state.terminal != (self.returncode is not None):
            raise ValueError("only terminal job states carry a return code")
        if self.state == JobState.SUCCEEDED and self.returncode != 0:
            raise ValueError("a succeeded job must have return code zero")
        if self.state == JobState.FAILED and self.returncode == 0:
            raise ValueError("a failed job cannot have return code zero")

    def to_json(self) -> str:
        return json.dumps(
            {
                "protocol": JOB_PROTOCOL_VERSION,
                "state": self.state.value,
                "pid": self.pid,
                "returncode": self.returncode,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "JobStatus":
        try:
            raw = json.loads(value)
            if not isinstance(raw, dict):
                raise TypeError
            if raw.get("protocol") != JOB_PROTOCOL_VERSION:
                raise ValueError("unsupported durable-job protocol version")
            state = JobState(raw["state"])
            pid = raw.get("pid")
            returncode = raw.get("returncode")
            if pid is not None and not isinstance(pid, int):
                raise TypeError
            if returncode is not None and not isinstance(returncode, int):
                raise TypeError
            return cls(state=state, pid=pid, returncode=returncode)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("invalid durable-job status") from exc


def launcher_script(command: str, paths: RemoteJobPaths, spec: JobSpec | None = None) -> str:
    """Build the detached wrapper which owns one remote command.

    ``command`` is a complete shell fragment produced by the SDK's existing
    quoted command builders. The wrapper, rather than an SSH channel, owns its
    descriptors and records its terminal status.
    """

    _validate_text(command, "remote command", empty=False)
    if spec is not None and spec.job_id != paths.job_id:
        raise ValueError("job specification and paths must use the same ID")
    job = shlex.quote(str(paths.directory))
    stdout = '"$job/stdout"' if spec is None or spec.stdout == "capture" else "/dev/null"
    stderr = '"$job/stderr"' if spec is None or spec.stderr == "capture" else "/dev/null"
    return f"""#!/bin/sh
set -u
umask 077
job={job}
status="$job/status.json"

write_status() {{
    state=$1
    code=$2
    temporary="$job/.status.$$"
    if [ "$code" = null ]; then
        printf '{{"pid":%s,"protocol":{JOB_PROTOCOL_VERSION},"returncode":null,"state":"%s"}}' "$$" "$state" >"$temporary"
    else
        printf '{{"pid":%s,"protocol":{JOB_PROTOCOL_VERSION},"returncode":%s,"state":"%s"}}' "$$" "$code" "$state" >"$temporary"
    fi
    mv -f -- "$temporary" "$status"
}}

terminate_job() {{
    code=$1
    trap - HUP INT TERM
    write_status terminated "$code"
    exit "$code"
}}

claim_candidate="$job/.execution.claim.$$"
printf '%s\n' "$$" >"$claim_candidate"
if ! ln -- "$claim_candidate" "$job/execution.claim" 2>/dev/null; then
    rm -f -- "$claim_candidate"
    exit 0
fi
rm -f -- "$claim_candidate"

write_status starting null
trap 'terminate_job 129' HUP
trap 'terminate_job 130' INT
trap 'terminate_job 143' TERM
write_status running null
set +e
(
    {command}
) </dev/null >{stdout} 2>{stderr}
code=$?
set -e
trap - HUP INT TERM
if [ -f "$job/termination.request" ]; then
    write_status terminated 143
elif [ "$code" -eq 0 ]; then
    write_status succeeded "$code"
else
    write_status failed "$code"
fi
exit 0
"""


def submission_command(
    spec: JobSpec,
    command: str,
    paths: RemoteJobPaths,
    *,
    staging_id: str | None = None,
) -> str:
    """Build an idempotent short SSH command which stages and starts a job.

    Linux ``mv -T`` atomically publishes the staged directory without nesting
    it when another submission of the same job won the race. Every retry stages
    identical content and the atomic execution-claim link admits one wrapper.
    """

    if paths.job_id != spec.job_id:
        raise ValueError("job specification and paths must use the same ID")
    nonce = validate_job_id(staging_id or new_job_id())
    root = shlex.quote(str(paths.root))
    job = shlex.quote(str(paths.directory))
    stage_path = paths.root / f".{paths.job_id}.{nonce}.stage"
    stage = shlex.quote(str(stage_path))
    encoded_spec = shlex.quote(spec.to_json())
    encoded_launcher = shlex.quote(launcher_script(command, paths, spec))
    prepared = shlex.quote(JobStatus(JobState.PREPARED).to_json())
    return f"""set -eu
umask 077
root={root}
job={job}
stage={stage}
cleanup() {{ rm -rf -- "$stage"; }}
trap cleanup EXIT HUP INT TERM
mkdir -p -- "$root"
chmod 700 -- "$root"
mkdir -- "$stage"
printf '%s' {encoded_spec} >"$stage/spec.json"
printf '%s' {encoded_launcher} >"$stage/launch.sh"
printf '%s' {prepared} >"$stage/status.json"
chmod 700 -- "$stage" "$stage/launch.sh"
: >"$stage/stdout"
: >"$stage/stderr"
: >"$stage/launcher.stderr"
if mv -T -- "$stage" "$job" 2>/dev/null; then
    :
else
    cmp -s -- "$stage/spec.json" "$job/spec.json"
    cmp -s -- "$stage/launch.sh" "$job/launch.sh"
fi
command -v flock >/dev/null
exec 9>"$job/submission.lock"
if ! flock -w 15 9; then
    echo 'durable job submission lock timed out' >&2
    exit 70
fi
start_launcher() {{
    if [ -e "$job/execution.claim" ]; then
        return
    fi
    command -v nohup >/dev/null
    command -v setsid >/dev/null
    nohup setsid sh "$job/launch.sh" </dev/null >/dev/null 2>"$job/launcher.stderr" &
}}

launch_round=0
while grep -q '\"state\":\"prepared\"' "$job/status.json"; do
    start_launcher
    attempt=0
    while grep -q '\"state\":\"prepared\"' "$job/status.json"; do
        if [ "$attempt" -ge 100 ]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if ! grep -q '\"state\":\"prepared\"' "$job/status.json"; then
        break
    fi
    claim_pid=$(cat -- "$job/execution.claim" 2>/dev/null || true)
    case "$claim_pid" in
        ''|*[!0-9]*) claim_live=false ;;
        *) if kill -0 "$claim_pid" 2>/dev/null; then claim_live=true; else claim_live=false; fi ;;
    esac
    if [ "$claim_live" = true ] || [ "$launch_round" -ge 1 ]; then
        echo 'durable job launcher did not acknowledge execution' >&2
        exit 70
    fi
    rm -rf -- "$job/execution.claim"
    launch_round=$((launch_round + 1))
done
cat -- "$job/status.json"
"""


__all__ = [
    "JOB_PROTOCOL_VERSION",
    "JOB_ROOT",
    "JobSpec",
    "JobState",
    "JobStatus",
    "OutputMode",
    "RemoteJobPaths",
    "can_transition",
    "launcher_script",
    "new_job_id",
    "submission_command",
    "validate_job_id",
]
