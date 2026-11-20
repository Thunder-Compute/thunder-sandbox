from __future__ import annotations

import asyncio
import io
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from unittest import mock

from thunder_sandbox._channels._spec import (
    DownloadRequest,
    ExecRequest,
    ExecState,
    ExecStatus,
    ExitReason,
    Identity,
    OutputMode,
    Retention,
    TerminationPolicy,
    TransferOptions,
    UploadRequest,
)
from thunder_sandbox._channels._ssh_channel import (
    SSHChannel,
    _AttachedHandle,
    _exec_status,
    _JobHandle,
)
from thunder_sandbox._common.exceptions import (
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    UnsupportedFeatureError,
)
from thunder_sandbox._common.types import (
    NetworkPolicy,
    Resources,
    SandboxInfo,
    SandboxStatus,
    SSHConnection,
)
from thunder_sandbox.asynchronous import _jobs
from thunder_sandbox.asynchronous._jobs import JobSpec, JobState, JobStatus, RemoteJobPaths
from thunder_sandbox.asynchronous._ssh import SSHConnectionManager
from thunder_sandbox.asynchronous.process import Process

EXEC_ID = "a" * 32


def info(**changes: object) -> SandboxInfo:
    fields: dict[str, object] = dict(
        id="sbx-test",
        name="test",
        status=SandboxStatus.READY,
        resources=Resources(cpu=4, memory=32, storage=50),
        network_policy=NetworkPolicy("open"),
        created_at=datetime.now(timezone.utc),
        ssh=SSHConnection("sandbox.example", 22, "root"),
        vm=True,
        image_id="image",
    )
    fields.update(changes)
    return SandboxInfo(**fields)  # type: ignore[arg-type]


def channel() -> SSHChannel:
    return SSHChannel(mock.Mock(), info)


class SupportTest(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_what_the_vm_cannot_honour(self) -> None:
        ssh = channel()
        refused = [
            ExecRequest(EXEC_ID, "true", umask=0o077),
            ExecRequest(EXEC_ID, "true", inherit_env=False),
            ExecRequest(EXEC_ID, "true", termination=TerminationPolicy(graceful_termination_timeout_ms=1)),
        ]
        for request in refused:
            with self.subTest(request=request), self.assertRaises(UnsupportedFeatureError):
                await ssh.exec(request)
        with self.assertRaises(UnsupportedFeatureError):
            await ssh.upload(UploadRequest(Path("x"), "/x", owner=Identity(user=0)))
        with self.assertRaises(UnsupportedFeatureError):
            await ssh.download(DownloadRequest("/x", Path("x"), TransferOptions(preserve_times=False)))

    async def test_accepts_its_fixed_values(self) -> None:
        ssh = channel()
        ssh._start_job = mock.AsyncMock(return_value="started")  # type: ignore[method-assign]
        result = await ssh.exec(ExecRequest(EXEC_ID, "true", umask=0o022, inherit_env=True))
        self.assertEqual(result, "started")


class UploadModeTest(unittest.IsolatedAsyncioTestCase):
    """Uploads keep each file's exact mode and times, as docker cp does."""

    def uploading_channel(self) -> tuple[SSHChannel, mock.AsyncMock]:
        ssh = channel()
        connection = object()

        async def run_transfer(transfer, *, name):  # noqa: ANN001
            await transfer(connection)

        ssh._retry_scp = run_transfer  # type: ignore[method-assign]
        ssh._reset_remote_path = mock.AsyncMock()  # type: ignore[method-assign]
        ssh._upload_target = mock.AsyncMock(return_value="/dest/file")  # type: ignore[method-assign]
        ssh._publish = mock.AsyncMock()  # type: ignore[method-assign]
        ssh._cleanup_remote_paths = mock.AsyncMock()  # type: ignore[method-assign]
        ssh._run = mock.AsyncMock()  # type: ignore[method-assign]
        return ssh, mock.AsyncMock()

    async def test_every_upload_scp_preserves_modes_and_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src"
            source.mkdir()
            (source / "file").write_text("x")
            for options in (
                TransferOptions(),
                TransferOptions(recursive=True),
                TransferOptions(recursive=True, contents_only=True),
            ):
                ssh, scp = self.uploading_channel()
                path = source / "file" if not options.recursive else source
                with self.subTest(options=options), mock.patch(
                    "thunder_sandbox._channels._ssh_channel.asyncssh.scp", scp
                ):
                    await ssh.upload(UploadRequest(path, "/dest", options))
                    self.assertTrue(scp.await_args_list)
                    for call in scp.await_args_list:
                        self.assertIs(call.kwargs.get("preserve"), True)

    async def test_upload_times_are_always_kept(self) -> None:
        ssh = channel()
        with self.assertRaises(UnsupportedFeatureError):
            await ssh.upload(UploadRequest(Path("x"), "/x", TransferOptions(preserve_times=False)))


class CommandTest(unittest.TestCase):
    def test_command_sets_umask_then_workdir_env_and_user(self) -> None:
        command = channel()._command(ExecRequest(
            EXEC_ID, "python", ("train.py", "a b"),
            working_dir="/work dir",
            env={"MODEL": "x y"},
            unset_env=("GONE",),
            identity=Identity(user="agent", group=1000),
        ))
        self.assertTrue(command.startswith("umask 0022; cd '/work dir' && "))
        argv = shlex.split(command.split("; ", 1)[1].split(" && ", 1)[1])
        self.assertEqual(argv[:2], ["sh", "-c"])
        self.assertEqual(argv[3:5], ["sh", "agent:1000"])
        self.assertEqual(argv[5:], ["env", "-u", "GONE", "MODEL=x y", "python", "train.py", "a b"])

    def test_command_without_workdir_starts_in_the_image_workdir(self) -> None:
        command = channel()._command(ExecRequest(EXEC_ID, "pwd"))
        self.assertTrue(command.startswith(
            "umask 0022; __thunder_workdir=$(cat /run/thunder_seed/vmroot/workdir"
        ))
        self.assertTrue(command.endswith("cd /; pwd"))


class StatusTest(unittest.TestCase):
    def test_job_states_map_to_exec_states(self) -> None:
        cases = [
            (JobStatus(JobState.PREPARED), ExecState.STARTING, None, None),
            (JobStatus(JobState.RUNNING, pid=7), ExecState.RUNNING, None, None),
            (JobStatus(JobState.SUCCEEDED, pid=7, returncode=0), ExecState.ENDED, ExitReason.EXITED, 0),
            (JobStatus(JobState.FAILED, pid=7, returncode=137), ExecState.ENDED, ExitReason.EXITED, 137),
            (JobStatus(JobState.TERMINATED, pid=7, returncode=143), ExecState.ENDED, ExitReason.TERMINATED, 143),
            (
                JobStatus(JobState.FAILED, returncode=137, interrupted=True),
                ExecState.ENDED, ExitReason.INTERRUPTED, None,
            ),
        ]
        for job, state, reason, returncode in cases:
            with self.subTest(job=job):
                status = _exec_status(EXEC_ID, job)
                self.assertEqual((status.state, status.reason, status.returncode), (state, reason, returncode))

    def test_interrupted_flag_is_invisible_to_older_readers(self) -> None:
        encoded = JobStatus(JobState.FAILED, returncode=137, interrupted=True).to_json()
        self.assertIn('"interrupted":true', encoded)
        self.assertEqual(JobStatus.from_json(encoded).interrupted, True)
        # What an SDK from before the flag reads: the same failed status.
        self.assertNotIn("interrupted", JobStatus(JobState.FAILED, returncode=137).to_json())
        with self.assertRaises(ValueError):
            JobStatus(JobState.FAILED, returncode=1, interrupted=True)


class FakeJobChannel:
    """The channel methods a job handle calls, scripted."""

    def __init__(self, statuses: list[JobStatus], output: dict[str, list[bytes]]) -> None:
        self.statuses = statuses
        self.output = {name: b"".join(chunks) for name, chunks in output.items()}
        self.chunk = 4
        self.signals: list[str] = []
        self.cleaned: list[str] = []

    async def _read_status(self, job_id: str, *, deadline: float | None) -> JobStatus:
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    async def _read_output(self, job_id: str, stream: str, offset: int) -> bytes:
        return self.output[stream][offset:offset + self.chunk]

    async def _signal(self, job_id: str, signal: str, pid: int) -> JobStatus:
        self.signals.append(signal)
        if signal == "KILL":
            self.statuses = [JobStatus(JobState.TERMINATED, pid=pid, returncode=137)]
        return self.statuses[0]

    async def _cleanup(self, job_id: str) -> None:
        self.cleaned.append(job_id)

    def _deadline(self) -> None:
        return None


def spec(**changes: object) -> JobSpec:
    return JobSpec(EXEC_ID, ("true",), **changes)  # type: ignore[arg-type]


class JobHandleTest(unittest.IsolatedAsyncioTestCase):
    async def test_streams_read_to_the_end_then_clean_up(self) -> None:
        fake = FakeJobChannel(
            [JobStatus(JobState.RUNNING, pid=7), JobStatus(JobState.SUCCEEDED, pid=7, returncode=0)],
            {"stdout": [b"hello world\n"], "stderr": [b""]},
        )
        handle = _JobHandle(fake, spec(), JobStatus(JobState.RUNNING, pid=7))  # type: ignore[arg-type]
        self.assertEqual(b"".join([chunk async for chunk in handle.stdout()]), b"hello world\n")
        self.assertEqual([chunk async for chunk in handle.stderr()], [])
        self.assertEqual((await handle.wait()).returncode, 0)
        await asyncio.sleep(0)
        self.assertEqual(fake.cleaned, [EXEC_ID])
        with self.assertRaises(RuntimeError):
            handle.stdout()

    async def test_retained_jobs_stay(self) -> None:
        fake = FakeJobChannel([JobStatus(JobState.SUCCEEDED, pid=7, returncode=0)], {"stdout": [], "stderr": []})
        handle = _JobHandle(fake, spec(retain=True), fake.statuses[0])  # type: ignore[arg-type]
        [chunk async for chunk in handle.stdout()]
        [chunk async for chunk in handle.stderr()]
        await handle.wait()
        await asyncio.sleep(0)
        self.assertEqual(fake.cleaned, [])
        await handle.cleanup()
        self.assertEqual(fake.cleaned, [EXEC_ID])

    async def test_terminate_escalates_to_sigkill(self) -> None:
        running = JobStatus(JobState.RUNNING, pid=7)
        fake = FakeJobChannel([running], {"stdout": [], "stderr": []})
        handle = _JobHandle(fake, spec(retain=True), running)  # type: ignore[arg-type]
        with mock.patch("thunder_sandbox._channels._ssh_channel.TERMINATION_GRACE_SECONDS", 0.05):
            status = await handle.terminate()
        self.assertEqual(fake.signals, ["TERM", "KILL"])
        self.assertEqual((status.reason, status.returncode), (ExitReason.TERMINATED, 137))

    async def test_a_failed_read_resumes_at_the_same_offset(self) -> None:
        fake = FakeJobChannel([JobStatus(JobState.SUCCEEDED, pid=7, returncode=0)],
                              {"stdout": [b"abcdefgh"], "stderr": []})
        reads: list[int] = []
        original = fake._read_output

        async def flaky(job_id: str, stream: str, offset: int) -> bytes:
            reads.append(offset)
            if offset == 4 and reads.count(4) == 1:
                raise ConnectionError("lost")
            return await original(job_id, stream, offset)

        fake._read_output = flaky  # type: ignore[method-assign]
        handle = _JobHandle(fake, spec(retain=True), fake.statuses[0])  # type: ignore[arg-type]
        stream = handle.stdout()
        self.assertEqual(await stream.__anext__(), b"abcd")
        with self.assertRaises(ConnectionError):
            await stream.__anext__()
        self.assertEqual(await stream.__anext__(), b"efgh")
        self.assertEqual(reads, [0, 4, 4])

    async def test_durable_jobs_have_no_stdin(self) -> None:
        fake = FakeJobChannel([JobStatus(JobState.RUNNING, pid=7)], {"stdout": [], "stderr": []})
        handle = _JobHandle(fake, spec(retain=True), fake.statuses[0])  # type: ignore[arg-type]
        with self.assertRaises(io.UnsupportedOperation):
            await handle.write_stdin(b"x")

    async def test_attach_refuses_a_bad_id(self) -> None:
        with self.assertRaises(InvalidRequestError):
            await channel().attach("../../etc")


class FakeReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        await asyncio.sleep(0)
        return self.chunks.pop(0) if self.chunks else b""


class AttachedHandleTest(unittest.IsolatedAsyncioTestCase):
    def process(self, **changes: object) -> mock.Mock:
        process = mock.Mock(**{"exit_status": None, "exit_signal": None, **changes})
        process.stdout = FakeReader([b"a", b"b"])
        process.stderr = FakeReader([])
        process.wait_closed = mock.AsyncMock()
        process.stdin.drain = mock.AsyncMock()
        return process

    async def test_drains_output_and_reports_exit(self) -> None:
        process = self.process()
        handle = _AttachedHandle(EXEC_ID, process)
        self.assertIs((await handle.poll()).state, ExecState.RUNNING)
        process.exit_status = 5
        self.assertEqual((await handle.wait()).returncode, 5)
        self.assertEqual(b"".join([chunk async for chunk in handle.stdout()]), b"ab")

    async def test_a_signal_reads_as_128_plus_its_number(self) -> None:
        process = self.process(exit_signal=("TERM", False, "", ""))
        self.assertEqual((await _AttachedHandle(EXEC_ID, process).wait()).returncode, 143)

    async def test_stdin_and_cleanup(self) -> None:
        process = self.process()
        handle = _AttachedHandle(EXEC_ID, process)
        await handle.write_stdin(b"x")
        process.stdin.write.assert_called_once_with(b"x")
        await handle.close_stdin()
        process.stdin.write_eof.assert_called_once()
        with self.assertRaises(io.UnsupportedOperation):
            await handle.cleanup()


class ScriptedHandle:
    exec_id = EXEC_ID

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.written: list[bytes | None] = []

    async def _stream(self, chunks: list[bytes]) -> AsyncIterator[bytes]:
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    def stdout(self) -> AsyncIterator[bytes]:
        return self._stream(self.chunks)

    def stderr(self) -> AsyncIterator[bytes]:
        return self._stream([])

    async def write_stdin(self, data: bytes) -> None:
        await asyncio.sleep(0)
        self.written.append(data)

    async def close_stdin(self) -> None:
        self.written.append(None)

    def last_status(self) -> None:
        return None

    async def wait(self) -> ExecStatus:
        return ExecStatus(EXEC_ID, ExecState.ENDED, reason=ExitReason.INTERRUPTED)


class HandleProcessTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_reads_decode_across_chunks(self) -> None:
        handle = ScriptedHandle([b"h\xc3", b"\xa9llo\nwor", b"ld"])
        process = Process.from_handle(handle, timeout=None, text=True, durable=True)  # type: ignore[arg-type]
        self.assertEqual(await process.stdout.readline(), "héllo\n")
        self.assertEqual(await process.stdout.read(2), "wo")
        self.assertEqual(await process.stdout.read(), "rld")
        self.assertEqual(await process.stdout.read(), "")

    async def test_bytes_lines_iterate(self) -> None:
        handle = ScriptedHandle([b"a\nb", b"\nc"])
        process = Process.from_handle(handle, timeout=None, text=False, durable=True)  # type: ignore[arg-type]
        self.assertEqual([line async for line in process.stdout], [b"a\n", b"b\n", b"c"])

    async def test_interrupted_reads_as_killed(self) -> None:
        process = Process.from_handle(ScriptedHandle([]), timeout=None, text=True, durable=True)  # type: ignore[arg-type]
        self.assertEqual(await process.wait(), 137)

    async def test_pty_writes_keep_their_order(self) -> None:
        handle = ScriptedHandle([])
        process = Process.from_handle(handle, timeout=None, text=True, durable=False)  # type: ignore[arg-type]
        process.stdin.write("a")
        process.stdin.write("b")
        process.stdin.write_eof()
        await process.stdin.drain()
        self.assertEqual(handle.written, [b"a", b"b", None])

    async def test_durable_processes_refuse_stdin(self) -> None:
        process = Process.from_handle(ScriptedHandle([]), timeout=None, text=True, durable=True)  # type: ignore[arg-type]
        with self.assertRaises(io.UnsupportedOperation):
            process.stdin.write("x")


# ------------------------------------------------------------ real job protocol


class _LocalCompleted:
    def __init__(self, returncode: int, stdout: object, stderr: object) -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class LocalConnection:
    """Runs a channel's SSH commands with this machine's shell, as a VM would."""

    def __init__(self) -> None:
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    async def run(self, command: str, *, input: str | None = None, check: bool = False,
                  encoding: str | None = "utf-8") -> _LocalCompleted:
        # A thread, not asyncio's subprocesses: the job launchers this starts
        # outlive the command, which confuses asyncio's child watcher.
        completed = await asyncio.to_thread(
            subprocess.run, ["sh", "-c", command],
            input=input.encode() if input is not None else None,
            capture_output=True, timeout=60,
        )
        stdout, stderr = completed.stdout, completed.stderr
        if encoding is not None:
            return _LocalCompleted(completed.returncode, stdout.decode(encoding), stderr.decode(encoding))
        return _LocalCompleted(completed.returncode, stdout, stderr)


HAS_JOB_TOOLS = all(shutil.which(tool) for tool in ("flock", "setsid", "nohup", "setpriv"))


@unittest.skipUnless(os.name == "posix" and HAS_JOB_TOOLS, "needs a Linux shell with util-linux")
class RealJobProtocolTest(unittest.IsolatedAsyncioTestCase):
    """Runs the real launcher, status, signal and output commands in a local shell."""

    async def asyncSetUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.boot_id = self.root / "boot_id"
        self.boot_id.write_text("boot-1\n")
        patches = [
            mock.patch.object(_jobs, "JOB_ROOT", PurePosixPath(self.root / "jobs")),
            mock.patch.object(_jobs, "_BOOT_ID", str(self.boot_id)),
            mock.patch.object(_jobs.RemoteJobPaths, "__init__", _paths_under(self.root / "jobs")),
            # The VM's image workdir file does not exist here; / is the fallback.
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.channel = SSHChannel(mock.Mock(), lambda: info(expires_at=None))
        connection = LocalConnection()
        self.channel._manager = SSHConnectionManager(mock.AsyncMock(return_value=connection))
        self.addCleanup(shutil.rmtree, self.root, True)

    def request(self, *argv: str, **changes: object) -> ExecRequest:
        return ExecRequest(os.urandom(16).hex(), argv[0], tuple(argv[1:]), **changes)  # type: ignore[arg-type]

    async def test_runs_reads_and_cleans_up(self) -> None:
        handle = await self.channel.exec(self.request(
            "sh", "-c", "printf out; printf err >&2; umask; exit 4", working_dir=str(self.root)
        ))
        process = Process.from_handle(handle, timeout=30, text=True, durable=True)
        self.assertEqual(await process.wait(), 4)
        self.assertEqual(await process.stdout.read(), "out0022\n")
        self.assertEqual(await process.stderr.read(), "err")
        await asyncio.sleep(0.5)
        self.assertFalse((self.root / "jobs" / handle.exec_id).exists())

    async def test_attach_finds_a_retained_job(self) -> None:
        request = self.request("sh", "-c", "sleep 0.3; echo later", retention=Retention.RETAIN)
        await self.channel.exec(request)
        again = await self.channel.attach(request.exec_id)
        self.assertEqual((await again.wait()).returncode, 0)
        self.assertEqual(b"".join([chunk async for chunk in again.stdout()]), b"later\n")
        await again.cleanup()
        with self.assertRaises(NotFoundError):
            await self.channel.attach(request.exec_id)

    async def test_terminate_stops_a_process_that_ignores_sigterm(self) -> None:
        handle = await self.channel.exec(self.request(
            "sh", "-c", "trap '' TERM; sleep 30", retention=Retention.RETAIN
        ))
        with mock.patch("thunder_sandbox._channels._ssh_channel.TERMINATION_GRACE_SECONDS", 0.5):
            status = await handle.terminate()
        self.assertEqual((status.reason, status.returncode), (ExitReason.TERMINATED, 137))
        await handle.cleanup()

    async def test_a_restart_interrupts_the_job(self) -> None:
        handle = await self.channel.exec(self.request("sleep", "30", retention=Retention.RETAIN))
        status = await handle.poll()
        self.assertIs(status.state, ExecState.RUNNING)
        # A new boot, with the job's supervisor gone as a reboot leaves it.
        pid = handle._status.pid  # type: ignore[attr-defined]
        os.killpg(pid, 9)
        self.boot_id.write_text("boot-2\n")
        await asyncio.sleep(0.3)
        status = await handle.wait()
        self.assertEqual((status.reason, status.returncode), (ExitReason.INTERRUPTED, None))
        await handle.cleanup()

    async def test_discarded_output_and_unknown_user(self) -> None:
        handle = await self.channel.exec(self.request(
            "echo", "hi", stdout=OutputMode.DISCARD, identity=Identity(user="no-such-user-x")
        ))
        process = Process.from_handle(handle, timeout=30, text=True, durable=True)
        self.assertEqual(await process.wait(), 1)
        self.assertEqual(await process.stdout.read(), "")
        self.assertIn("unable to find user", await process.stderr.read())


def _paths_under(root: Path):  # type: ignore[no-untyped-def]
    original = RemoteJobPaths.__init__

    def init(self, job_id: str, root_: PurePosixPath = PurePosixPath(root)) -> None:  # type: ignore[no-untyped-def]
        original(self, job_id, PurePosixPath(root))

    return init


if __name__ == "__main__":
    unittest.main()


class SharedConnectionTest(unittest.IsolatedAsyncioTestCase):
    """One connection serves every operation, so one session's failure keeps it."""

    def manager(self, *connections: object) -> tuple[SSHConnectionManager, mock.AsyncMock]:
        opened = mock.AsyncMock(side_effect=list(connections))
        manager = SSHConnectionManager(opened, sleep=mock.AsyncMock(), jitter=lambda _a, _b: 0.0)
        return manager, opened

    async def test_a_refused_session_keeps_the_connection(self) -> None:
        import asyncssh

        live = mock.Mock(is_closed=mock.Mock(return_value=False))
        manager, opened = self.manager(live)
        attempts = 0

        async def operation(connection: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncssh.ChannelOpenError(asyncssh.OPEN_RESOURCE_SHORTAGE, "too many sessions")
            return "done"

        self.assertEqual(await manager.run(operation, name="op"), "done")
        self.assertEqual(opened.await_count, 1)
        live.close.assert_not_called()

    async def test_any_error_on_a_connection_that_closed_is_retried(self) -> None:
        state = {"closed": False}
        first = mock.Mock(is_closed=mock.Mock(side_effect=lambda: state["closed"]))
        second = mock.Mock(is_closed=mock.Mock(return_value=False))
        manager, opened = self.manager(first, second)

        async def operation(connection: object) -> str:
            if connection is first:
                state["closed"] = True
                raise AssertionError("AsyncSSH opened a session on a closed connection")
            return "done"

        self.assertEqual(await manager.run(operation, name="op"), "done")
        self.assertEqual(opened.await_count, 2)

    async def test_an_unrelated_error_on_a_live_connection_is_raised(self) -> None:
        live = mock.Mock(is_closed=mock.Mock(return_value=False))
        manager, _ = self.manager(live)

        async def operation(connection: object) -> str:
            raise AssertionError("a bug, not a lost connection")

        with self.assertRaises(AssertionError):
            await manager.run(operation, name="op")
