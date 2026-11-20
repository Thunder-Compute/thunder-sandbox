"""Shell fragments and SSH helpers shared by Sandbox and the SSH channels.

Everything here builds what runs on the sandbox, or reads what comes back; none
of it holds connection state.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import asyncssh

from .._common.exceptions import SandboxFailedError
from .._common.types import SSHConnection
from ._jobs import container_job_directory, validate_job_id
from ._ssh import RetryableSSHOperationError
from ._transfer import (
    ByteChannel as _ByteChannel,
    extract_download_archive as _extract_download_archive,
)

_CONTAINER_NAME = "thunder-sandbox"
_CONTAINER_BUSYBOX = "/busybox"

def _known_host_name(ssh: SSHConnection) -> str:
    return ssh.host if ssh.port == 22 else f"[{ssh.host}]:{ssh.port}"
# Host keys are remembered only for the life of this process. A sandbox is
# short-lived and nothing verifies the key on the first connection, so a file
# would add staleness and an unwritable-home failure mode without buying any
# trust the process does not already have.
_REMEMBERED_HOST_KEYS: dict[str, str] = {}


def _host_key_text(key: object) -> str:
    if hasattr(key, "export_public_key"):
        exported = key.export_public_key()  # type: ignore[union-attr]
        text = exported.decode("ascii") if isinstance(exported, bytes) else str(exported)
    else:
        text = str(key)
    text = text.strip()
    if not text:
        raise SandboxFailedError("Thunder returned an empty SSH host key")
    try:
        return asyncssh.import_public_key(text).export_public_key().decode("ascii").strip()
    except (asyncssh.Error, UnicodeError, ValueError) as exc:
        raise SandboxFailedError("Thunder returned an invalid SSH host key") from exc


def _pinned_host_key(sandbox_id: str) -> "asyncssh.SSHKey | None":
    text = _REMEMBERED_HOST_KEYS.get(sandbox_id)
    if text is None:
        return None
    try:
        return asyncssh.import_public_key(text)
    except (asyncssh.Error, UnicodeError, ValueError):
        return None


def _remember_host_key(sandbox_id: str, key: object) -> None:
    _REMEMBERED_HOST_KEYS[sandbox_id] = _host_key_text(key)



# Archives one download source to stdout. $1 is the source, $2 and $3 are 1
# for a recursive and a contents-only download, and $4 prefixes tar: the
# container's BusyBox, or empty for the guest's own. tar -h follows every
# symlink where it is archived, as SCP did, so the archive holds only files and
# directories: a link never reaches the caller's machine, where its target
# would name the caller's own files. A dangling link or a loop fails the
# download, as it did over SCP.
_DOWNLOAD_ARCHIVE_SCRIPT = """source=$1 recursive=$2 contents=$3 tool=$4
while [ "$source" != / ] && [ "${source%/}" != "$source" ]; do
    source=${source%/}
done
if [ ! -e "$source" ]; then
    echo "$1: No such file or directory" >&2
    exit 2
fi
if [ -d "$source" ] && [ "$recursive" != 1 ]; then
    echo "$1: is a directory; download it with recursive=True" >&2
    exit 2
fi
if [ "$contents" = 1 ]; then
    if [ ! -d "$source" ]; then
        echo "$1: Not a directory" >&2
        exit 2
    fi
    cd -- "$source" || exit 2
    exec $tool tar -c -h -f - .
fi
case $source in
*/*) parent=${source%/*} name=${source##*/} ;;
*) parent=. name=$source ;;
esac
cd -- "${parent:-/}" || exit 2
exec $tool tar -c -h -f - -- "$name"
"""
_DOWNLOAD_CHUNK_BYTES = 1 << 16
_DOWNLOAD_STDERR_LIMIT = 4096


def _download_archive_command(
    source: str, *, recursive: bool, contents_only: bool, container: bool
) -> str:
    flags = ("1" if recursive else "0", "1" if contents_only else "0")
    if container:
        # docker cp reads a relative container path from /, not the image's
        # WORKDIR, and uploads still go through docker cp: keep them agreeing.
        if not source.startswith("/"):
            source = "/" + source
        # As root in the container, as docker cp was: a verifier's root-only
        # output must still download. BusyBox is bind-mounted into every
        # image, so the image needs no tar of its own.
        args: tuple[str, ...] = (
            "sudo",
            "--non-interactive",
            "docker",
            "exec",
            "--user",
            "0",
            _CONTAINER_NAME,
            _CONTAINER_BUSYBOX,
            "sh",
            "-c",
            _DOWNLOAD_ARCHIVE_SCRIPT,
            "sh",
            source,
            *flags,
            _CONTAINER_BUSYBOX,
        )
    else:
        args = ("sh", "-c", _DOWNLOAD_ARCHIVE_SCRIPT, "sh", source, *flags, "")
    return " ".join(shlex.quote(arg) for arg in args)


async def _stream_download_archive(
    connection: asyncssh.SSHClientConnection,
    command: str,
    destination: Path,
    *,
    single_root: bool,
    name: str,
) -> str | None:
    """Run ``command`` and extract the tar archive it writes into ``destination``.

    A lost connection raises an ``OSError`` so the caller's SSH manager can
    reconnect and restart the download; any other failure is final.
    """

    channel = _ByteChannel()
    reader = channel.reader()

    def extract() -> str | None:
        try:
            return _extract_download_archive(
                reader, destination, single_root=single_root
            )
        finally:
            # Stops the SSH side waiting on a queue nobody reads any more.
            channel.abort()

    extractor = asyncio.ensure_future(asyncio.to_thread(extract))
    process: asyncssh.SSHClientProcess[bytes] | None = None
    stderr: asyncio.Future[bytes] | None = None
    try:
        try:
            process = await connection.create_process(
                command, encoding=None, stdin=asyncssh.DEVNULL
            )
        except (OSError, asyncssh.Error) as exc:
            if connection.is_closed():
                raise RetryableSSHOperationError(f"{name} could not start: {exc}") from exc
            raise SandboxFailedError(f"could not start {name}: {exc}") from exc
        stderr = asyncio.ensure_future(process.stderr.read())

        stream_error: BaseException | None = None
        exit_status: int | None = None
        try:
            delivering = True
            while True:
                chunk = await process.stdout.read(_DOWNLOAD_CHUNK_BYTES)
                if delivering and not await asyncio.to_thread(
                    channel.put, chunk or None
                ):
                    # The extractor is finished. After a complete archive
                    # tar may still send padding, which is read and dropped
                    # so the exit status arrives; after a failure, stop.
                    delivering = False
                    await asyncio.wait({extractor})
                    if extractor.exception() is not None:
                        process.close()
                        break
                if not chunk:
                    exit_status = (await process.wait()).exit_status
                    break
        except (OSError, asyncssh.Error) as exc:
            stream_error = exc
        channel.abort()
        await asyncio.wait({extractor})
        extract_error = extractor.exception()

        if connection.is_closed():
            raise RetryableSSHOperationError(
                f"{name} lost its connection"
            ) from (stream_error or extract_error)
        if stream_error is not None:
            raise SandboxFailedError(
                f"could not complete {name}: {stream_error}"
            ) from stream_error
        detail = await _stderr_detail(stderr)
        if exit_status not in (None, 0):
            raise SandboxFailedError(
                f"could not complete {name}: the archive exited with status "
                f"{exit_status}" + (f": {detail}" if detail else "")
            )
        if extract_error is not None:
            raise SandboxFailedError(
                f"could not complete {name}: {extract_error}"
            ) from extract_error
        if exit_status is None:
            raise SandboxFailedError(
                f"could not complete {name}: the archive ended without an exit status"
                + (f": {detail}" if detail else "")
            )
        return extractor.result()
    finally:
        channel.abort()
        if process is not None:
            process.close()
        if stderr is not None and not stderr.done():
            stderr.cancel()
        # The extractor thread stops once the channel is aborted. Wait for it
        # so a retry or the cleanup never races its writes into the stage.
        await asyncio.wait({extractor})
        if not extractor.cancelled():
            extractor.exception()


async def _stderr_detail(stderr: asyncio.Future[bytes]) -> str:
    if not stderr.done():
        await asyncio.wait({stderr}, timeout=1.0)
    if not stderr.done() or stderr.cancelled() or stderr.exception() is not None:
        return ""
    text = bytes(stderr.result()).decode("utf-8", "replace").strip()
    return text[-_DOWNLOAD_STDERR_LIMIT:]


def _remote_command(
    args: Sequence[str],
    *,
    workdir: str | None,
    env: Mapping[str, str | None] | None,
    user: str | None = None,
) -> str:
    parts: list[str] = []
    if workdir is not None:
        parts.extend(("cd", shlex.quote(workdir), "&&"))
    if user is not None:
        parts.extend(("sh", "-c", shlex.quote(_SWITCH_USER_SCRIPT), "sh", shlex.quote(user)))
    if env:
        # env takes its options before any assignment: after one, "-u" would
        # be the command to run.
        parts.append("env")
        for key, value in env.items():
            if value is None:
                parts.extend(("-u", shlex.quote(key)))
        for key, value in env.items():
            if value is not None:
                parts.append(f"{shlex.quote(key)}={shlex.quote(value)}")
    parts.extend(shlex.quote(arg) for arg in args)
    return " ".join(parts)


# Enters a VM sandbox's image working directory, or / when the sandbox names
# none (an empty name included) or the directory no longer exists.
_VM_IMAGE_WORKDIR = (
    '__thunder_workdir=$(cat /run/thunder_seed/vmroot/workdir 2>/dev/null); '
    'cd -- "${__thunder_workdir:-/}" 2>/dev/null || cd /; '
)


# Runs "$@" as $1, which is a user or UID with an optional ":group" or ":GID",
# as ``docker exec --user`` takes it: a named user gets its primary group, its
# supplementary groups and its home; an explicit group replaces the
# supplementary groups; a UID without an account gets group 0 and HOME=/, as in
# Docker, and an unknown user or group fails with status 1 and Docker's
# message. setpriv execs in place, so the command keeps the caller's process
# group and its exit status, and environment apart from HOME, USER and LOGNAME.
_SWITCH_USER_SCRIPT = """set -e
__thunder_spec=$1
shift
__thunder_user=${__thunder_spec%%:*}
__thunder_group=
case $__thunder_spec in
*:*) __thunder_group=${__thunder_spec#*:} ;;
esac
if [ -z "$__thunder_user" ]; then
    echo "invalid user: $__thunder_spec" >&2
    exit 1
fi
if __thunder_entry=$(getent passwd "$__thunder_user"); then
    IFS=: read -r __thunder_name _ __thunder_uid __thunder_gid _ __thunder_home _ <<__thunder_eof
$__thunder_entry
__thunder_eof
    __thunder_groups=--init-groups
else
    case $__thunder_user in
    *[!0-9]*)
        echo "unable to find user $__thunder_user: no matching entries in passwd file" >&2
        exit 1
        ;;
    esac
    __thunder_name=$__thunder_user __thunder_uid=$__thunder_user __thunder_gid=0 __thunder_home=/
    __thunder_groups=--clear-groups
fi
if [ -n "$__thunder_group" ]; then
    if __thunder_entry=$(getent group "$__thunder_group"); then
        IFS=: read -r _ _ __thunder_gid _ <<__thunder_eof
$__thunder_entry
__thunder_eof
    else
        case $__thunder_group in
        ''|*[!0-9]*)
            echo "unable to find group $__thunder_group: no matching entries in group file" >&2
            exit 1
            ;;
        esac
        __thunder_gid=$__thunder_group
    fi
    __thunder_groups=--clear-groups
fi
exec setpriv --reuid="$__thunder_uid" --regid="$__thunder_gid" "$__thunder_groups" -- \
    env HOME="$__thunder_home" USER="$__thunder_name" LOGNAME="$__thunder_name" "$@"
"""


# Names the wrapper sets in the shell that execs the payload. Anything it sets
# there would overwrite an inherited variable of the same name, so they are
# prefixed. The search runs in a subshell and leaves IFS and globbing alone.
_CONTAINER_RESOLVE_COMMAND = """__thunder_name=$1
shift
__thunder_path=$(
    set -f
    IFS=:
    case "$__thunder_name" in
    (*/*) exit 0 ;;
    esac
    for directory in ${PATH:-}; do
        candidate=${directory:-.}/$__thunder_name
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s' "$candidate"
            exit 0
        fi
    done
)
"""
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _container_exec_script(unset: Sequence[str], container_busybox: str) -> str:
    """Exec the payload, preferring the image's program to a BusyBox applet.

    BusyBox's shell prefers its own applets to PATH, even for ``exec``, so a bare
    ``sh`` or ``ls`` would run BusyBox rather than the image's program. A path
    with a slash is always executed as written, so the command is looked up in
    PATH first, and ``exec -a`` keeps the name the caller used as ``argv[0]``.
    BusyBox remains the fallback for images which lack the command altogether.

    ``unset`` in this shell aborts it on a name which isn't a valid identifier,
    so ``env -u`` removes those, at the cost of ``argv[0]`` becoming the path.
    """

    script = _CONTAINER_RESOLVE_COMMAND
    invalid = [key for key in unset if not _SHELL_NAME.fullmatch(key)]
    script += "".join(f"unset {key}\n" for key in unset if key not in invalid)
    if invalid:
        removals = " ".join(f"-u {shlex.quote(key)}" for key in invalid)
        return (
            script
            + f"exec {shlex.quote(container_busybox)} env {removals} "
            + '"${__thunder_path:-$__thunder_name}" "$@"\n'
        )
    return (
        script
        + 'if [ -n "$__thunder_path" ]; then\n'
        + '    exec -a "$__thunder_name" "$__thunder_path" "$@"\n'
        + "fi\n"
        + 'exec "$__thunder_name" "$@"\n'
    )


def _container_session_waiter(container_busybox: str) -> str:
    """Start ``setsid`` as a child and wait for the session leader it execs.

    A ``setsid`` process which is already a process-group leader must fork
    before it can create a new session. Docker may start its exec process in
    exactly that state. Invoking ``setsid`` directly would then let the parent
    exit zero while the real command continued unobserved. A background child
    of this non-interactive shell inherits the shell's process group and cannot
    itself be its leader, so ``setsid`` can exec the payload in place. Waiting
    here keeps ``docker exec`` attached to the payload and preserves its code.
    """

    busybox = shlex.quote(container_busybox)
    return f"""set +e
{busybox} setsid "$@" &
session_pid=$!
wait "$session_pid"
code=$?
exit "$code"
"""


def _container_job_prepare_script(job_root: str, busybox: str) -> str:
    """Create job parents without following attacker-controlled symlinks.

    Each verified directory protects the next component from replacement.
    BusyBox stat uses lstat by default; do not add its symlink-following -L.
    """

    root = PurePosixPath(job_root)
    tmp = shlex.quote(str(root.parent.parent))
    parent = shlex.quote(str(root.parent))
    jobs = shlex.quote(str(root))
    binary = shlex.quote(busybox)
    return f"""set -eu
umask 022
# A root-owned sticky /tmp protects root-owned entries from replacement.
# A root-owned 0755 /tmp is safe as well.
case "$({binary} stat -c '%u:%a:%F' -- {tmp})" in
    '0:1777:directory'|'0:755:directory') ;;
    *) echo 'unsafe container temporary directory' >&2; exit 70 ;;
esac
{binary} mkdir -m 755 -- {parent} 2>/dev/null || :
[ "$({binary} stat -c '%u:%a:%F' -- {parent})" = '0:755:directory' ] || {{
    echo 'unsafe container job parent' >&2
    exit 70
}}
{binary} mkdir -m 1777 -- {jobs} 2>/dev/null || :
[ "$({binary} stat -c '%u:%a:%F' -- {jobs})" = '0:1777:directory' ] || {{
    echo 'unsafe container job directory' >&2
    exit 70
}}
"""


def _container_remote_command(
    args: Sequence[str],
    *,
    workdir: str | None,
    env: Mapping[str, str | None] | None,
    pty: bool,
    job_id: str | None = None,
    user: str | None = None,
) -> str:
    parts = ["sudo", "--non-interactive", "docker", "exec", "--interactive"]
    if pty:
        parts.append("--tty")
    if user is not None:
        parts.extend(("--user", user))
    if workdir is not None:
        parts.extend(("--workdir", workdir))
    unset: list[str] = []
    for key, value in (env or {}).items():
        if value is None:
            unset.append(key)
        else:
            parts.extend(("--env", f"{key}={value}"))
    parts.append(_CONTAINER_NAME)
    if job_id is None:
        if unset:
            parts.extend((_CONTAINER_BUSYBOX, "env"))
            for key in unset:
                parts.extend(("-u", key))
        parts.extend(args)
    else:
        validate_job_id(job_id)
        container_job = str(container_job_directory(job_id))
        inner = (
            "set -eu\n"
            f"__thunder_job={shlex.quote(container_job)}\n"
            f'{shlex.quote(_CONTAINER_BUSYBOX)} mkdir -m 700 -- "$__thunder_job"\n'
            'printf \'%s\\n\' "$$" >"$__thunder_job/pid"\n'
            + _container_exec_script(unset, _CONTAINER_BUSYBOX)
        )
        # Do not make setsid the docker-exec process. If that process is already
        # a process-group leader, BusyBox forks and lets the observed parent
        # return zero before the payload finishes. The waiter launches setsid
        # as its non-leader child and propagates the payload's actual status.
        parts.extend(
            (
                _CONTAINER_BUSYBOX,
                "sh",
                "-c",
                _container_session_waiter(_CONTAINER_BUSYBOX),
                "sh",
                _CONTAINER_BUSYBOX,
                "sh",
                "-c",
                inner,
                "sh",
            )
        )
        parts.extend(args)
    command = " ".join(shlex.quote(part) for part in parts)
    if job_id is not None:
        prepare = _container_job_prepare_script(
            str(container_job_directory(job_id).parent), _CONTAINER_BUSYBOX
        )
        # Job payloads can use different users within the same container.
        # Prepare the shared parent as root, then let the selected user own
        # its job directory. UID 0 works even without a root passwd entry.
        # Lifecycle operations also execute as UID 0.
        command = (
            "sudo --non-interactive docker exec --user 0 "
            f"{shlex.quote(_CONTAINER_NAME)} {shlex.quote(_CONTAINER_BUSYBOX)} "
            f"sh -c {shlex.quote(prepare)} && {command}"
        )
    return command

