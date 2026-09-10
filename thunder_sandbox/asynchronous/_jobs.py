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
from typing import Mapping

from .._common.types import OutputMode

JOB_PROTOCOL_VERSION = 1
JOB_ROOT = PurePosixPath("/var/tmp/thunder-sandbox/jobs")
CONTAINER_JOB_ROOT = PurePosixPath("/tmp/thunder-sandbox/jobs")

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
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
    """Stable host paths owned by one job.

    A caller may override ``root`` in tests, but production paths are fixed and
    never contain user-controlled path components other than a validated ID.
    Ephemeral files whose names contain a shell PID stay launcher-local.
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
    def status(self) -> PurePosixPath:
        return self.directory / "status.json"

    @property
    def supervisor_pid(self) -> PurePosixPath:
        return self.directory / "supervisor.pid"

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
    def submission_lock(self) -> PurePosixPath:
        return self.directory / "submission.lock"

    @property
    def run_lock(self) -> PurePosixPath:
        return self.directory / "run.lock"

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
    """Immutable command metadata staged for recovery and operator debugging.

    Recovery consumes the output, retention, and container fields. The argv,
    workdir, and environment fields preserve the submitted command's intent for
    inspection; execution itself uses the separately persisted launch script.
    """

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


_STATUS_PID_PLACEHOLDER = 987654321
_STATUS_CODE_PLACEHOLDER = 123


def _state_pattern(*states: JobState) -> str:
    return "|".join(f'"state":"{state.value}"' for state in states)


def _status_printf_format(
    state: JobState, returncode: int | None = None, *, dynamic_code: bool = False
) -> str:
    code = _STATUS_CODE_PLACEHOLDER if dynamic_code else returncode
    encoded = JobStatus(
        state, pid=_STATUS_PID_PLACEHOLDER, returncode=code
    ).to_json()
    encoded = encoded.replace(f'"pid":{_STATUS_PID_PLACEHOLDER}', '"pid":%s')
    if dynamic_code:
        encoded = encoded.replace(
            f'"returncode":{_STATUS_CODE_PLACEHOLDER}', '"returncode":%s'
        )
    return encoded


def container_job_directory(job_id: str) -> PurePosixPath:
    """Return the private in-container state directory for a durable job."""

    return CONTAINER_JOB_ROOT / validate_job_id(job_id)


def launcher_script(command: str, paths: RemoteJobPaths, spec: JobSpec) -> str:
    """Build the detached wrapper which owns one remote command.

    ``command`` is a complete shell fragment produced by the SDK's existing
    quoted command builders. The wrapper, rather than an SSH channel, owns its
    descriptors and records its terminal status.
    """

    _validate_text(command, "remote command", empty=False)
    if spec.job_id != paths.job_id:
        raise ValueError("job specification and paths must use the same ID")
    job = shlex.quote(str(paths.directory))
    status = shlex.quote(str(paths.status))
    run_lock = shlex.quote(str(paths.run_lock))
    supervisor_pid = shlex.quote(str(paths.supervisor_pid))
    termination_request = shlex.quote(str(paths.termination_request))
    stdout = '"$job/stdout"' if spec.stdout == "capture" else "/dev/null"
    stderr = '"$job/stderr"' if spec.stderr == "capture" else "/dev/null"
    prepared_pattern = shlex.quote(_state_pattern(JobState.PREPARED))
    starting_format = shlex.quote(_status_printf_format(JobState.STARTING))
    running_format = shlex.quote(_status_printf_format(JobState.RUNNING))
    succeeded_format = shlex.quote(
        _status_printf_format(JobState.SUCCEEDED, returncode=0)
    )
    failed_format = shlex.quote(
        _status_printf_format(JobState.FAILED, dynamic_code=True)
    )
    terminated_format = shlex.quote(
        _status_printf_format(JobState.TERMINATED, returncode=143)
    )
    return f"""#!/bin/sh
set -u
umask 077
job={job}
status={status}
exec 8>{run_lock}
if ! flock -n 8; then
    exit 0
fi
if ! grep -Eq {prepared_pattern} "$status"; then
    exit 0
fi

write_status() {{
    state=$1
    code=$2
    temporary="$job/.status.$$"
    case "$state" in
        starting) printf {starting_format} "$$" >"$temporary" ;;
        running) printf {running_format} "$$" >"$temporary" ;;
        succeeded) printf {succeeded_format} "$$" >"$temporary" ;;
        failed) printf {failed_format} "$$" "$code" >"$temporary" ;;
        terminated) printf {terminated_format} "$$" >"$temporary" ;;
        *) exit 64 ;;
    esac
    mv -f -- "$temporary" "$status"
}}

payload_group_running() {{
    for process_stat in /proc/[0-9]*/stat; do
        [ -r "$process_stat" ] || continue
        IFS= read -r process_record <"$process_stat" || continue
        process_pid=${{process_record%% *}}
        process_fields=${{process_record##*) }}
        set -- $process_fields
        if [ "$process_pid" != "$$" ] && [ "${{3:-}}" = "$$" ]; then
            return 0
        fi
    done
    return 1
}}

pid_temporary="$job/.supervisor.pid.$$"
printf '%s\n' "$$" >"$pid_temporary"
mv -f -- "$pid_temporary" {supervisor_pid}
write_status starting null
# A signal can interrupt the shell's wait even when the payload ignores it.
# Keep the wrapper alive until the payload really exits so status never gets
# ahead of the process group and terminate() can enforce its SIGKILL grace.
trap ':' HUP INT TERM
write_status running null
set +e
(
    {command}
) </dev/null >{stdout} 2>{stderr} &
payload_pid=$!
while :; do
    wait "$payload_pid"
    code=$?
    if kill -0 "$payload_pid" 2>/dev/null; then
        continue
    fi
    break
done
set -e
trap - HUP INT TERM
if [ -f {termination_request} ]; then
    # The immediate subshell may have died from SIGTERM while one of its
    # descendants ignored the signal. Do not publish terminal state until the
    # whole job process group is actually gone; terminate() will escalate it.
    while payload_group_running; do
        sleep 0.05
    done
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
    identical content and the lifetime run lock admits one wrapper.
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
    prepared_pattern = shlex.quote(_state_pattern(JobState.PREPARED))
    submission_lock = shlex.quote(str(paths.submission_lock))
    status = shlex.quote(str(paths.status))
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
exec 9>{submission_lock}
if ! flock -w 15 9; then
    echo 'durable job submission lock timed out' >&2
    exit 70
fi
start_launcher() {{
    command -v nohup >/dev/null
    command -v setsid >/dev/null
    nohup setsid sh "$job/launch.sh" </dev/null >/dev/null 2>"$job/launcher.stderr" &
}}

launch_round=0
while grep -Eq {prepared_pattern} {status}; do
    start_launcher
    attempt=0
    while grep -Eq {prepared_pattern} {status}; do
        if [ "$attempt" -ge 100 ]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.05
    done
    if ! grep -Eq {prepared_pattern} {status}; then
        break
    fi
    if [ "$launch_round" -ge 1 ]; then
        echo 'durable job launcher did not acknowledge execution' >&2
        exit 70
    fi
    launch_round=$((launch_round + 1))
done
cat -- {status}
"""


def cleanup_command(
    paths: RemoteJobPaths,
    *,
    container: str | None = None,
    container_busybox: str = "/busybox",
) -> str:
    """Build an idempotent command which removes every durable-job artifact."""

    container_cleanup = ""
    if container is not None:
        container_cleanup = (
            "sudo --non-interactive docker exec "
            f"{shlex.quote(container)} {shlex.quote(container_busybox)} rm -rf -- "
            f"{shlex.quote(str(container_job_directory(paths.job_id)))} "
            "2>/dev/null || true\n"
        )
    return container_cleanup + f"rm -rf -- {shlex.quote(str(paths.directory))}"


@dataclass(frozen=True)
class _ProcessGroupTarget:
    prepare: str
    send_signal: str
    is_alive: str


def _host_process_group(signal: str) -> _ProcessGroupTarget:
    return _ProcessGroupTarget(
        prepare="",
        send_signal=f'kill -{signal} -- "-$expected_pid" 2>/dev/null || true',
        is_alive='kill -0 -- "-$expected_pid" 2>/dev/null',
    )


def _container_process_group(
    paths: RemoteJobPaths,
    *,
    signal: str,
    container: str,
    container_busybox: str,
    terminal_pattern: str,
) -> _ProcessGroupTarget:
    quoted_container = shlex.quote(container)
    quoted_busybox = shlex.quote(container_busybox)
    quoted_pid_file = shlex.quote(
        str(container_job_directory(paths.job_id) / "pid")
    )
    prepare = f"""
container_pid=''
attempt=0
while [ -z "$container_pid" ]; do
    container_pid=$(sudo --non-interactive docker exec {quoted_container} {quoted_busybox} cat -- {quoted_pid_file} 2>/dev/null || true)
    if [ -n "$container_pid" ] || grep -Eq {terminal_pattern} "$status"; then
        break
    fi
    if [ "$attempt" -ge 40 ]; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if grep -Eq {terminal_pattern} "$status"; then
    cat -- "$status"
    exit 0
fi
case "$container_pid" in
    ''|*[!0-9]*)
        echo 'durable container job PID is unavailable' >&2
        exit 70
        ;;
esac
"""
    prefix = (
        f"sudo --non-interactive docker exec {quoted_container} {quoted_busybox} kill"
    )
    return _ProcessGroupTarget(
        prepare=prepare,
        send_signal=f'{prefix} -{signal} -- "-$container_pid" 2>/dev/null || true',
        is_alive=f'{prefix} -0 -- "-$container_pid" 2>/dev/null',
    )


def signal_command(
    paths: RemoteJobPaths,
    *,
    signal: str,
    pid: int,
    container: str | None = None,
    container_busybox: str = "/busybox",
) -> str:
    """Build an idempotent process-group signal and status reconciliation."""

    if signal not in {"TERM", "KILL"}:
        raise ValueError(f"invalid durable job signal: {signal!r}")
    if pid <= 0:
        raise ValueError("durable job PID must be positive")
    job = shlex.quote(str(paths.directory))
    terminal_pattern = shlex.quote(
        _state_pattern(*(state for state in JobState if state.terminal))
    )
    terminated_status = shlex.quote(
        JobStatus(JobState.TERMINATED, pid=pid, returncode=137).to_json()
    )
    target = (
        _host_process_group(signal)
        if container is None
        else _container_process_group(
            paths,
            signal=signal,
            container=container,
            container_busybox=container_busybox,
            terminal_pattern=terminal_pattern,
        )
    )
    kill_completion = ""
    if signal == "KILL":
        kill_completion = f"""
attempt=0
while {target.is_alive}; do
    if [ "$attempt" -ge 40 ]; then
        exit 70
    fi
    attempt=$((attempt + 1))
    sleep 0.05
done
if grep -Eq {terminal_pattern} "$status"; then
    cat -- "$status"
    exit 0
fi
temporary="$job/.status.terminate.$$"
printf '%s' {terminated_status} >"$temporary"
mv -f -- "$temporary" "$status"
"""
    return f"""set -eu
job={job}
status={shlex.quote(str(paths.status))}
touch -- {shlex.quote(str(paths.termination_request))}
if grep -Eq {terminal_pattern} "$status"; then
    cat -- "$status"
    exit 0
fi
expected_pid={pid}
recorded_pid=$(cat -- {shlex.quote(str(paths.supervisor_pid))} 2>/dev/null || true)
if [ "$recorded_pid" != "$expected_pid" ]; then
    echo 'durable job PID does not match its supervisor record' >&2
    exit 64
fi
{target.prepare}
{target.send_signal}
{kill_completion}
cat -- "$status"
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
    "cleanup_command",
    "container_job_directory",
    "launcher_script",
    "new_job_id",
    "signal_command",
    "submission_command",
    "validate_job_id",
]
