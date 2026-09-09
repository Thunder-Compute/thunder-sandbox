from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import subprocess
import tempfile
import tarfile
from datetime import datetime, timezone
import time
import threading
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

import asyncssh

import thunder_sandbox as thunder
import thunder_sandbox.asynchronous as asynchronous
import thunder_sandbox.synchronous as synchronous
from thunder_sandbox._common.config import DEFAULT_API_URL, ClientConfig, ThunderPaths
from thunder_sandbox._common.exceptions import (
    AuthenticationError,
    CapacityError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    SandboxError,
    SandboxFailedError,
    SandboxTimeoutError,
    ServiceUnavailableError,
    _WaitWindowElapsedError,
)
from thunder_sandbox._common.types import GPUType, SandboxStatus
from thunder_sandbox.asynchronous.client import USER_AGENT, _api_error
from thunder_sandbox.asynchronous.client import Client as AsyncClient
from thunder_sandbox.asynchronous.process import Process as AsyncProcess
from thunder_sandbox.asynchronous._jobs import (
    JOB_PROTOCOL_VERSION,
    JobSpec,
    JobState,
    JobStatus,
    RemoteJobPaths,
    can_transition,
    launcher_script,
    new_job_id,
    submission_command,
)
from thunder_sandbox.asynchronous._ssh import (
    SSH_CONNECT_TIMEOUT_SECONDS,
    SSHConnectionManager,
)
from thunder_sandbox.asynchronous.sandbox import WAIT_WINDOW_MAX_SECONDS
from thunder_sandbox.asynchronous.sandbox import Sandbox as AsyncSandbox
from thunder_sandbox.asynchronous.sandbox import _pinned_host_key
from thunder_sandbox.image import Image, ResolvedImage, _create_canonical_build_context
from thunder_sandbox.synchronous._bridge import AsyncBridge
from thunder_sandbox.synchronous.client import Client
from thunder_sandbox.synchronous.process import Process
from thunder_sandbox.synchronous.sandbox import Sandbox
from test.ssh_faults import FakeSSHConnection, SSHDisconnected, SSHFaults, disconnect

SANDBOX_RESPONSE = {
    "id": "sbx-test",
    "name": "worker",
    "status": "ready",
    "spec": {"cpu_count": 4, "memory_gib": 32, "storage_gib": 50},
    "network_policy": {
        "internet_access": "restricted",
        "cidr_allowlist": ["0.0.0.0/0"],
        "domain_allowlist": ["*"],
    },
    "created_at": "2026-08-23T12:00:00Z",
    "expires_at": "2099-08-23T13:00:00Z",
    "ssh": {"host": "sandbox.example", "port": 2222, "user": "ubuntu"},
}


def config(directory: str) -> ClientConfig:
    return ClientConfig(
        api_url="https://api.example",
        api_token="token",
        paths=ThunderPaths(Path(directory)),
    )


def prepare_key(paths: ThunderPaths) -> None:
    paths.sandbox_keys.mkdir(parents=True, exist_ok=True)
    paths.sandbox_private_key("sbx-test").write_text("PRIVATE", encoding="utf-8")


def canonical_context(root: Path):
    return _create_canonical_build_context(
        root, root / ".thunder" / "image_build_contexts"
    )


def still_starting() -> _WaitWindowElapsedError:
    """What the wait endpoint answers when its window closes on a starting sandbox."""
    return _WaitWindowElapsedError(
        "Sandbox is still starting. Retry the wait request.",
        code="sandbox_wait_timeout",
        status=408,
        retry_after=0,
    )


def route_missing() -> NotFoundError:
    """What an API that predates the wait endpoint answers, indistinguishable
    from a missing sandbox."""
    return NotFoundError(
        "The requested resource was not found", code="not_found", status=404
    )


class DurableJobProtocolTest(unittest.IsolatedAsyncioTestCase):
    def test_job_ids_are_random_and_safe_remote_path_components(self) -> None:
        first = new_job_id()
        second = new_job_id()
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first, second)
        paths = RemoteJobPaths(first)
        self.assertEqual(paths.directory.name, first)
        self.assertEqual(paths.status, paths.directory / "status.json")
        for unsafe in ("", ".", "../escape", "A" * 32, "0" * 31):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                RemoteJobPaths(unsafe)

    def test_job_spec_is_structured_versioned_and_deterministic(self) -> None:
        job_id = "a" * 32
        environment = {"MODEL": "large", "DEBUG": None}
        spec = JobSpec(
            job_id,
            ("python", "train.py", "value with spaces"),
            workdir="/workspace",
            env=environment,
            container="thunder-sandbox",
        )
        environment["MODEL"] = "mutated"
        encoded = spec.to_json()
        self.assertEqual(encoded, spec.to_json())
        decoded = json.loads(encoded)
        self.assertEqual(decoded["protocol"], JOB_PROTOCOL_VERSION)
        self.assertEqual(decoded["argv"], list(spec.argv))
        self.assertEqual(decoded["env"], {"MODEL": "large", "DEBUG": None})
        self.assertEqual(JobSpec.from_json(encoded), spec)

        legacy = dict(decoded)
        legacy.pop("stdout")
        legacy.pop("stderr")
        legacy.pop("retain")
        recovered = JobSpec.from_json(json.dumps(legacy))
        self.assertEqual((recovered.stdout, recovered.stderr), ("capture", "capture"))
        self.assertFalse(recovered.retain)

    def test_job_spec_rejects_values_unsafe_at_the_remote_boundary(self) -> None:
        job_id = "b" * 32
        for kwargs in (
            {"argv": ()},
            {"argv": ("echo\x00bad",)},
            {"argv": ("echo",), "workdir": ""},
            {"argv": ("echo",), "env": {"BAD=NAME": "value"}},
            {"argv": ("echo",), "env": {"GOOD": "bad\x00value"}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JobSpec(job_id, **kwargs)

    def test_job_state_machine_and_status_records_are_strict(self) -> None:
        self.assertTrue(can_transition(JobState.PREPARED, JobState.STARTING))
        self.assertTrue(can_transition(JobState.PREPARED, JobState.SUCCEEDED))
        self.assertTrue(can_transition(JobState.RUNNING, JobState.SUCCEEDED))
        self.assertFalse(can_transition(JobState.RUNNING, JobState.PREPARED))
        self.assertFalse(can_transition(JobState.SUCCEEDED, JobState.RUNNING))

        running = JobStatus(JobState.RUNNING, pid=123)
        self.assertEqual(JobStatus.from_json(running.to_json()), running)
        completed = JobStatus(JobState.SUCCEEDED, pid=123, returncode=0)
        self.assertEqual(JobStatus.from_json(completed.to_json()), completed)
        with self.assertRaises(ValueError):
            JobStatus(JobState.RUNNING, returncode=0)
        with self.assertRaises(ValueError):
            JobStatus(JobState.SUCCEEDED, returncode=1)
        with self.assertRaisesRegex(ValueError, "protocol version"):
            JobStatus.from_json('{"protocol":2,"state":"running"}')

    async def test_ssh_fault_plan_disconnects_at_exact_occurrences(self) -> None:
        faults = SSHFaults(
            {
                "status": {2: disconnect("status connection lost")},
                "stdout": {1: disconnect("output connection lost")},
            }
        )
        await faults.checkpoint("status")
        with self.assertRaisesRegex(SSHDisconnected, "status connection lost"):
            await faults.checkpoint("status")
        await faults.checkpoint("status")
        with self.assertRaisesRegex(SSHDisconnected, "output connection lost"):
            await faults.checkpoint("stdout")
        self.assertEqual(faults.hits("status"), 3)
        self.assertEqual(faults.hits("stdout"), 1)

    def test_submission_is_detached_and_reuses_one_execution_claim(self) -> None:
        job_id = "c" * 32
        paths = RemoteJobPaths(job_id)
        spec = JobSpec(job_id, ("echo", "hello"))
        command = submission_command(
            spec,
            "echo hello",
            paths,
            staging_id="d" * 32,
        )
        self.assertIn('nohup setsid sh "$job/launch.sh"', command)
        self.assertIn("</dev/null >/dev/null", command)
        self.assertIn(
            'ln -- "$claim_candidate" "$job/execution.claim"',
            launcher_script("echo hello", paths),
        )
        self.assertIn('claim_pid=$(cat -- "$job/execution.claim"', command)
        self.assertIn('kill -0 "$claim_pid"', command)
        self.assertIn('rm -rf -- "$job/execution.claim"', command)
        self.assertIn('flock -w 15 9', command)
        self.assertIn("exit 70", command)
        self.assertNotIn("exit 75", command)
        self.assertIn("trap 'terminate_job 143' TERM", launcher_script("echo hello", paths))
        self.assertIn('if [ -f "$job/termination.request" ]', launcher_script("echo hello", paths))
        self.assertEqual(command.count("nohup setsid"), 1)
        syntax = subprocess.run(
            ["sh", "-n"], input=command, text=True, capture_output=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    async def test_launcher_executes_a_payload_at_most_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_id = "e" * 32
            paths = RemoteJobPaths(job_id, root=PurePosixPath(directory))
            Path(paths.directory).mkdir(mode=0o700)
            side_effect = root / "side-effect"
            command = f"printf x >> {shlex.quote(str(side_effect))}"
            script = Path(paths.launcher)
            script.write_text(launcher_script(command, paths), encoding="utf-8")
            Path(paths.status).write_text(
                JobStatus(JobState.PREPARED).to_json(), encoding="utf-8"
            )
            Path(paths.stdout).touch()
            Path(paths.stderr).touch()

            first = await asyncio.create_subprocess_exec("sh", str(script))
            second = await asyncio.create_subprocess_exec("sh", str(script))
            self.assertEqual(await first.wait(), 0)
            self.assertEqual(await second.wait(), 0)

            self.assertEqual(side_effect.read_text(encoding="utf-8"), "x")
            status = JobStatus.from_json(Path(paths.status).read_bytes())
            self.assertEqual(status.state, JobState.SUCCEEDED)
            self.assertEqual(status.returncode, 0)

    async def test_launcher_redirects_output_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_id = "2" * 32
            paths = RemoteJobPaths(job_id, root=PurePosixPath(directory))
            Path(paths.directory).mkdir(mode=0o700)
            Path(paths.launcher).write_text(
                launcher_script(
                    "printf output; printf error >&2; exit 7", paths
                ),
                encoding="utf-8",
            )
            Path(paths.status).write_text(
                JobStatus(JobState.PREPARED).to_json(), encoding="utf-8"
            )

            process = await asyncio.create_subprocess_exec(
                "sh", str(paths.launcher)
            )
            self.assertEqual(await process.wait(), 0)

            self.assertEqual(Path(paths.stdout).read_text(encoding="utf-8"), "output")
            self.assertEqual(Path(paths.stderr).read_text(encoding="utf-8"), "error")
            status = JobStatus.from_json(Path(paths.status).read_bytes())
            self.assertEqual(status.state, JobState.FAILED)
            self.assertEqual(status.returncode, 7)

    def test_launcher_can_discard_either_output_stream(self) -> None:
        job_id = "3" * 32
        paths = RemoteJobPaths(job_id)
        spec = JobSpec(job_id, ("command",), stdout="discard", stderr="capture")
        script = launcher_script("command", paths, spec)
        self.assertIn(") </dev/null >/dev/null 2>\"$job/stderr\"", script)


class SSHConnectionManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_healthy_connection_is_reused(self) -> None:
        connection = FakeSSHConnection()
        opened = mock.AsyncMock(return_value=connection)
        manager = SSHConnectionManager(opened)  # type: ignore[arg-type]

        async def identity(candidate):
            return candidate

        self.assertIs(await manager.run(identity, name="first"), connection)
        self.assertIs(await manager.run(identity, name="second"), connection)
        opened.assert_awaited_once_with()
        await manager.close()

    async def test_a_transport_failure_reconnects_and_retries(self) -> None:
        first = FakeSSHConnection()
        second = FakeSSHConnection()
        opened = mock.AsyncMock(side_effect=[first, second])
        slept = mock.AsyncMock()
        manager = SSHConnectionManager(
            opened,  # type: ignore[arg-type]
            sleep=slept,
            jitter=lambda _start, _end: 0.0,
        )
        faults = SSHFaults({"status": {1: disconnect()}})

        async def status(connection):
            await faults.checkpoint("status")
            return connection

        self.assertIs(await manager.run(status, name="status"), second)
        self.assertTrue(first.closed)
        self.assertFalse(second.closed)
        self.assertEqual(opened.await_count, 2)
        slept.assert_awaited_once_with(0.25)
        await manager.close()

    async def test_concurrent_callers_share_one_connection_attempt(self) -> None:
        connection = FakeSSHConnection()
        release = asyncio.Event()

        async def open_connection():
            await release.wait()
            return connection

        opened = mock.AsyncMock(side_effect=open_connection)
        manager = SSHConnectionManager(opened)  # type: ignore[arg-type]
        first = asyncio.create_task(manager.get())
        second = asyncio.create_task(manager.get())
        await asyncio.sleep(0)
        release.set()
        self.assertEqual(await asyncio.gather(first, second), [connection, connection])
        opened.assert_awaited_once_with()
        await manager.close()

    async def test_discarding_an_old_connection_keeps_its_replacement(self) -> None:
        old = FakeSSHConnection()
        replacement = FakeSSHConnection()
        manager = SSHConnectionManager(  # type: ignore[arg-type]
            mock.AsyncMock(return_value=replacement)
        )
        manager._connection = replacement  # type: ignore[assignment]

        await manager.discard(old)  # type: ignore[arg-type]

        self.assertIs(await manager.get(), replacement)
        self.assertTrue(old.closed)
        self.assertFalse(replacement.closed)
        await manager.close()

    async def test_retry_deadline_bounds_a_persistent_outage(self) -> None:
        now = [10.0]

        async def sleep(delay: float) -> None:
            now[0] += delay

        opened = mock.AsyncMock(side_effect=OSError("network down"))
        manager = SSHConnectionManager(
            opened,  # type: ignore[arg-type]
            sleep=sleep,
            clock=lambda: now[0],
            jitter=lambda _start, _end: 0.0,
        )
        with self.assertRaisesRegex(ConnectionError, "SSH retry deadline"):
            await manager.run(
                mock.AsyncMock(), name="bounded operation", deadline=10.3
            )
        self.assertEqual(opened.await_count, 3)

    async def test_authentication_failure_is_not_retried(self) -> None:
        opened = mock.AsyncMock(
            side_effect=asyncssh.PermissionDenied("certificate rejected")
        )
        manager = SSHConnectionManager(opened)  # type: ignore[arg-type]
        with self.assertRaises(asyncssh.PermissionDenied):
            await manager.run(mock.AsyncMock(), name="authenticate")
        opened.assert_awaited_once_with()


class DurableProcessTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def output_reader(values: dict[str, bytes]):
        async def read_output(stream: str, offset: int, size: int) -> bytes:
            return values[stream][offset : offset + size]

        return read_output

    async def test_poll_and_wait_follow_persistent_status(self) -> None:
        statuses = iter(
            [
                JobStatus(JobState.RUNNING, pid=101),
                JobStatus(JobState.SUCCEEDED, pid=101, returncode=0),
            ]
        )
        deadlines: list[float | None] = []

        async def read_status(deadline: float | None) -> JobStatus:
            deadlines.append(deadline)
            return next(statuses)

        process = AsyncProcess.durable(
            "4" * 32,
            status=JobStatus(JobState.STARTING, pid=101),
            read_status=read_status,
            read_output=self.output_reader({"stdout": b"result\n", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        self.assertEqual(process.id, "4" * 32)
        self.assertTrue(process.is_durable)
        self.assertIsNone(await process.poll())
        self.assertEqual(await process.wait(), 0)
        self.assertEqual(process.returncode, 0)
        self.assertIsNotNone(deadlines[0])
        self.assertIsNone(deadlines[1])

    async def test_incremental_output_remains_available_after_wait(self) -> None:
        output_reads: list[tuple[str, int, int]] = []

        async def read_status(_deadline: float | None) -> JobStatus:
            raise AssertionError("terminal status should stay cached")

        values = {"stdout": b"first\nsecond\n", "stderr": b"warning"}

        async def read_output(stream: str, offset: int, size: int) -> bytes:
            output_reads.append((stream, offset, size))
            return values[stream][offset : offset + size]

        cleanup = mock.AsyncMock()

        process = AsyncProcess.durable(
            "5" * 32,
            status=JobStatus(JobState.SUCCEEDED, pid=202, returncode=0),
            read_status=read_status,
            read_output=read_output,
            cleanup_job=cleanup,
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        self.assertEqual(await process.wait(), 0)
        self.assertEqual(await process.stdout.readline(), "first\n")
        self.assertEqual(await process.stdout.read(3), "sec")
        self.assertEqual(await process.stdout.read(), "ond\n")
        self.assertEqual(await process.stderr.read(), "warning")
        await asyncio.sleep(0)
        self.assertEqual(
            [(stream, offset) for stream, offset, _size in output_reads],
            [("stdout", 0), ("stdout", 13), ("stderr", 0), ("stderr", 7)],
        )
        cleanup.assert_awaited_once_with()

    async def test_exit_flush_during_status_read_is_not_mistaken_for_eof(self) -> None:
        values = {"stdout": b"", "stderr": b""}

        async def read_status(_deadline: float | None) -> JobStatus:
            # Reproduce a payload flushing its final bytes while the terminal
            # status SSH round trip is in flight.
            values["stdout"] = b"final output line\n"
            return JobStatus(JobState.SUCCEEDED, pid=202, returncode=0)

        process = AsyncProcess.durable(
            "9" * 32,
            status=JobStatus(JobState.RUNNING, pid=202),
            read_status=read_status,
            read_output=self.output_reader(values),
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        self.assertEqual(await process.stdout.read(), "final output line\n")

    async def test_byte_cursor_retries_without_duplicate_output(self) -> None:
        value = "start-🧁-finish\n".encode()
        attempts: list[int] = []
        failed = False

        async def read_output(_stream: str, offset: int, _size: int) -> bytes:
            nonlocal failed
            attempts.append(offset)
            if offset == 6 and not failed:
                failed = True
                raise SSHDisconnected("lost before chunk reached client")
            # Deliberately split the four-byte code point across reads.
            widths = {0: 6, 6: 2, 8: 2}
            width = widths.get(offset, 3)
            return value[offset : offset + width]

        process = AsyncProcess.durable(
            "a" * 32,
            status=JobStatus(JobState.SUCCEEDED, pid=202, returncode=0),
            read_status=mock.AsyncMock(),
            read_output=read_output,
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        with self.assertRaises(SSHDisconnected):
            await process.stdout.read()
        self.assertEqual(await process.stdout.read(), "start-🧁-finish\n")
        self.assertEqual(attempts[:3], [0, 6, 6])

    async def test_discarded_stream_is_empty_and_does_not_touch_remote_file(self) -> None:
        read_output = mock.AsyncMock(side_effect=AssertionError("must not read"))
        cleanup = mock.AsyncMock()
        process = AsyncProcess.durable(
            "b" * 32,
            status=JobStatus(JobState.SUCCEEDED, pid=202, returncode=0),
            read_status=mock.AsyncMock(),
            read_output=read_output,
            cleanup_job=cleanup,
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=False,
            stdout="discard",
            stderr="discard",
        )

        self.assertEqual(await process.stdout.read(), b"")
        self.assertEqual(await process.stderr.read(), b"")
        await asyncio.sleep(0)
        read_output.assert_not_awaited()
        cleanup.assert_awaited_once_with()

    async def test_retain_suppresses_automatic_cleanup(self) -> None:
        cleanup = mock.AsyncMock()
        process = AsyncProcess.durable(
            "c" * 32,
            status=JobStatus(JobState.SUCCEEDED, pid=202, returncode=0),
            read_status=mock.AsyncMock(),
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=cleanup,
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
            retain=True,
        )

        await process.stdout.read()
        await process.stderr.read()
        cleanup.assert_not_awaited()
        await process.cleanup()
        await process.cleanup()
        cleanup.assert_awaited_once_with()

    async def test_durable_stdin_is_explicitly_unsupported(self) -> None:
        async def read_status(_deadline: float | None) -> JobStatus:
            return JobStatus(JobState.RUNNING, pid=303)

        process = AsyncProcess.durable(
            "6" * 32,
            status=JobStatus(JobState.RUNNING, pid=303),
            read_status=read_status,
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        with self.assertRaises(io.UnsupportedOperation):
            process.stdin.write("input")
        with self.assertRaises(io.UnsupportedOperation):
            await process.stdin.drain()
        with self.assertRaises(io.UnsupportedOperation):
            process.stdin.write_eof()

    async def test_durable_terminate_records_a_terminal_status(self) -> None:
        signal = mock.AsyncMock(
            return_value=JobStatus(JobState.TERMINATED, pid=303, returncode=143)
        )
        process = AsyncProcess.durable(
            "6" * 32,
            status=JobStatus(JobState.RUNNING, pid=303),
            read_status=mock.AsyncMock(),
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=signal,
            timeout=None,
            text=True,
        )

        await process.terminate()

        signal.assert_awaited_once_with("TERM", 303)
        self.assertEqual(process.returncode, 143)
        self.assertEqual(await process.wait(), 143)

    async def test_durable_terminate_escalates_to_kill(self) -> None:
        signal = mock.AsyncMock(
            side_effect=[
                JobStatus(JobState.RUNNING, pid=404),
                JobStatus(JobState.TERMINATED, pid=404, returncode=137),
            ]
        )
        process = AsyncProcess.durable(
            "d" * 32,
            status=JobStatus(JobState.RUNNING, pid=404),
            read_status=mock.AsyncMock(),
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=signal,
            timeout=None,
            text=True,
        )

        with mock.patch(
            "thunder_sandbox.asynchronous.process.PROCESS_TERMINATION_GRACE_SECONDS",
            0.0,
        ):
            await process.terminate()

        self.assertEqual(
            [call.args for call in signal.await_args_list],
            [("TERM", 404), ("KILL", 404)],
        )
        self.assertEqual(process.returncode, 137)

    async def test_concurrent_durable_termination_is_idempotent(self) -> None:
        signal = mock.AsyncMock(
            return_value=JobStatus(JobState.TERMINATED, pid=505, returncode=143)
        )
        process = AsyncProcess.durable(
            "e" * 32,
            status=JobStatus(JobState.RUNNING, pid=505),
            read_status=mock.AsyncMock(),
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=signal,
            timeout=None,
            text=True,
        )

        await asyncio.gather(process.terminate(), process.terminate())

        signal.assert_awaited_once_with("TERM", 505)

    async def test_wait_timeout_does_not_change_remote_status(self) -> None:
        calls = 0

        async def read_status(_deadline: float | None) -> JobStatus:
            nonlocal calls
            calls += 1
            return JobStatus(JobState.RUNNING, pid=404)

        process = AsyncProcess.durable(
            "7" * 32,
            status=JobStatus(JobState.RUNNING, pid=404),
            read_status=read_status,
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        with self.assertRaises(asyncio.TimeoutError):
            await process.wait(timeout=0.01)
        self.assertIsNone(process.returncode)
        self.assertGreaterEqual(calls, 1)

    async def test_concurrent_waiters_share_terminal_status(self) -> None:
        calls = 0

        async def read_status(_deadline: float | None) -> JobStatus:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return JobStatus(JobState.SUCCEEDED, pid=505, returncode=0)

        process = AsyncProcess.durable(
            "8" * 32,
            status=JobStatus(JobState.RUNNING, pid=505),
            read_status=read_status,
            read_output=self.output_reader({"stdout": b"", "stderr": b""}),
            cleanup_job=mock.AsyncMock(),
            signal_job=mock.AsyncMock(),
            timeout=None,
            text=True,
        )

        self.assertEqual(await asyncio.gather(process.wait(), process.wait()), [0, 0])
        self.assertEqual(calls, 1)


class ConfigTest(unittest.TestCase):
    def test_public_distribution_exports_both_apis(self) -> None:
        self.assertIs(thunder.Client, Client)
        self.assertIs(thunder.Sandbox, Sandbox)
        self.assertIs(thunder.Process, Process)
        self.assertIs(thunder.Image, Image)
        self.assertIs(thunder.ResolvedImage, ResolvedImage)
        self.assertIs(synchronous.Client, Client)
        self.assertIs(synchronous.Sandbox, Sandbox)
        self.assertIs(synchronous.Process, Process)
        self.assertIs(asynchronous.Client, AsyncClient)
        self.assertIs(asynchronous.Sandbox, AsyncSandbox)
        self.assertIs(asynchronous.Process, AsyncProcess)
        self.assertIs(asynchronous.Image, Image)
        self.assertIs(synchronous.Image, Image)
        self.assertEqual(AsyncClient.__name__, Client.__name__)
        self.assertEqual(AsyncSandbox.__name__, Sandbox.__name__)
        self.assertEqual(AsyncProcess.__name__, Process.__name__)
        self.assertEqual(set(asynchronous.__all__), set(synchronous.__all__))
        self.assertIs(asynchronous.GPUType, synchronous.GPUType)
        self.assertEqual(USER_AGENT, f"thunder-python-sdk/{thunder.__version__}")

    def test_public_io_methods_have_async_twins(self) -> None:
        for cls, methods in {
            thunder.Client: (
                "close", "create_sandbox", "get_sandbox",
                "get_sandbox_by_name", "list_sandboxes", "resolve_image",
            ),
            thunder.Sandbox: (
                "create",
                "download",
                "exec",
                "from_id",
                "from_name",
                "get_process",
                "poll",
                "refresh",
                "terminate",
                "update_network_policy",
                "upload",
                "wait",
                "wait_until_ready",
            ),
            thunder.Process: ("cleanup", "poll", "terminate", "wait"),
        }.items():
            for method in methods:
                with self.subTest(cls=cls.__name__, method=method):
                    self.assertTrue(callable(getattr(cls, f"{method}_async")))

    def test_configuration_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ThunderPaths(root / "state")
            paths.root.mkdir()
            paths.credentials.write_text(
                json.dumps({"token": "file", "api_url": "https://file"}),
                encoding="utf-8",
            )
            (root / ".thunder.json").write_text(
                json.dumps({"api_url": "https://project"}), encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {"TNR_API_TOKEN": "environment", "TNR_API_URL": "https://environment"},
                clear=True,
            ):
                resolved = ClientConfig(paths=paths)
            self.assertEqual(resolved.api_token, "environment")
            self.assertEqual(resolved.api_url, "https://environment")

    def test_project_config_cannot_redirect_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ThunderPaths(root / "state")
            paths.root.mkdir()
            paths.credentials.write_text(
                json.dumps({"token": "secret", "api_url": "https://api.example"}),
                encoding="utf-8",
            )
            (root / ".thunder.json").write_text(
                json.dumps({"api_url": "https://attacker.example"}), encoding="utf-8"
            )
            with mock.patch("thunder_sandbox._common.config.Path.cwd", return_value=root):
                resolved = ClientConfig(paths=paths)
            self.assertEqual(resolved.api_url, "https://api.example")

    def test_api_url_requires_https(self) -> None:
        with self.assertRaisesRegex(InvalidRequestError, "HTTPS"):
            ClientConfig(api_url="http://api.example", api_token="secret")

    def test_default_url_and_missing_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            resolved = ClientConfig(paths=ThunderPaths(Path(directory)))
            self.assertEqual(resolved.api_url, DEFAULT_API_URL)
            with self.assertRaises(AuthenticationError):
                AsyncClient(resolved)

    def test_paths_reject_unsafe_ids(self) -> None:
        paths = ThunderPaths(Path("/tmp/thunder-test"))
        for value in ("", ".", "..", "a/b", "a\\b"):
            with self.subTest(value=value), self.assertRaises(InvalidRequestError):
                paths.sandbox_private_key(value)


class ImageTest(unittest.TestCase):
    def test_registry_image_accepts_public_or_complete_private_credentials(
        self,
    ) -> None:
        public = Image.from_registry("ubuntu:24.04")
        private = Image.from_registry(
            "registry.example.com/team/image:latest",
            username="user",
            password="secret",
            display_name="Training image",
        )

        self.assertEqual(repr(public), "Image.from_registry('ubuntu:24.04')")
        self.assertNotIn("secret", repr(private))
        self.assertIn("<redacted>", repr(private))

    def test_registry_image_rejects_invalid_definitions(self) -> None:
        for url, username, password in (
            ("", None, None),
            ("image with spaces", None, None),
            ("private.example/image", "user", None),
            ("private.example/image", None, "password"),
            ("private.example/image", "", "password"),
            ("private.example/image", "user", ""),
        ):
            with self.subTest(url=url, username=username, password=password):
                with self.assertRaises(InvalidRequestError):
                    Image.from_registry(url, username=username, password=password)

    def test_dockerfile_image_requires_a_context_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = Path(directory)
            with self.assertRaises(InvalidRequestError):
                Image.from_dockerfile(context)

            (context / "Dockerfile").write_text(
                "FROM ubuntu:24.04\n",
                encoding="utf-8",
            )
            image = Image.from_dockerfile(context)
            self.assertEqual(
                repr(image), f"Image.from_dockerfile({str(context.resolve())!r})"
            )

    def test_image_display_name_must_be_trimmed_single_line_and_bounded(self) -> None:
        for display_name in (" image", "image\nname", "é" * 65):
            with self.subTest(display_name=display_name):
                with self.assertRaises(InvalidRequestError):
                    Image.from_registry("ubuntu:24.04", display_name=display_name)

    def test_dockerfile_image_rejects_a_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidRequestError):
                Image.from_dockerfile(Path(directory) / "missing")

    def test_canonical_context_ignores_filesystem_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\nCOPY payload /payload\n")
            payload = root / "payload"
            payload.write_bytes(b"same bytes\n")
            first = canonical_context(root)
            try:
                first_bytes = first.archive_path.read_bytes()
                os.chmod(payload, 0o755)
                os.utime(payload, (1_000_000_000, 1_100_000_000))
                second = canonical_context(root)
                try:
                    self.assertEqual(first.context_hash, second.context_hash)
                    self.assertEqual(first.recipe_hash, second.recipe_hash)
                    self.assertEqual(first_bytes, second.archive_path.read_bytes())
                    with tarfile.open(second.archive_path, "r:") as archive:
                        for member in archive.getmembers():
                            self.assertEqual(member.mtime, 0)
                            self.assertEqual(member.uid, 0)
                            self.assertEqual(member.gid, 0)
                            self.assertEqual(member.mode, 0o644)
                finally:
                    second.close()
            finally:
                first.close()

    def test_canonical_context_hashes_paths_and_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            first_path = root / "first"
            first_path.write_bytes(b"payload")
            first = canonical_context(root)
            try:
                first_path.rename(root / "second")
                renamed = canonical_context(root)
                try:
                    self.assertNotEqual(first.context_hash, renamed.context_hash)
                finally:
                    renamed.close()
                (root / "second").write_bytes(b"different")
                changed = canonical_context(root)
                try:
                    self.assertNotEqual(first.context_hash, changed.context_hash)
                finally:
                    changed.close()
            finally:
                first.close()

    def test_canonical_context_applies_dockerignore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            (root / ".dockerignore").write_text("*.log\n!important.log\n")
            (root / "ignored.log").write_bytes(b"ignored")
            (root / "important.log").write_bytes(b"included")
            nested = root / "nested"
            nested.mkdir()
            (nested / "included.log").write_bytes(b"included by Docker semantics")
            context = canonical_context(root)
            try:
                with tarfile.open(context.archive_path, "r:") as archive:
                    self.assertEqual(
                        [member.name for member in archive.getmembers()],
                        [
                            ".dockerignore",
                            "Dockerfile",
                            "important.log",
                            "nested/included.log",
                        ],
                    )
            finally:
                context.close()

    def test_canonical_context_does_not_inspect_ignored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            (root / ".dockerignore").write_text(
                "node_modules\nignored-link\nignored-pipe\n"
            )
            ignored_directory = root / "node_modules"
            ignored_directory.mkdir()
            try:
                (ignored_directory / "link").symlink_to(root / "Dockerfile")
                (root / "ignored-link").symlink_to(root / "Dockerfile")
            except (NotImplementedError, OSError):
                self.skipTest("symbolic links are unavailable")
            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "ignored-pipe")

            scanned_directories: list[Path] = []
            scandir = os.scandir

            def record_scandir(path: str | os.PathLike[str]):
                scanned_directories.append(Path(path))
                return scandir(path)

            with mock.patch(
                "thunder_sandbox.image.os.scandir", side_effect=record_scandir
            ):
                context = canonical_context(root)
            try:
                self.assertNotIn(ignored_directory, scanned_directories)
                with tarfile.open(context.archive_path, "r:") as archive:
                    self.assertEqual(
                        [member.name for member in archive.getmembers()],
                        [".dockerignore", "Dockerfile"],
                    )
            finally:
                context.close()

    def test_canonical_context_descends_for_negated_ignored_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            (root / ".dockerignore").write_text(
                "generated\n!generated/included.txt\n"
            )
            generated = root / "generated"
            generated.mkdir()
            (generated / "excluded.txt").write_bytes(b"excluded")
            (generated / "included.txt").write_bytes(b"included")

            context = canonical_context(root)
            try:
                with tarfile.open(context.archive_path, "r:") as archive:
                    self.assertEqual(
                        [member.name for member in archive.getmembers()],
                        [
                            ".dockerignore",
                            "Dockerfile",
                            "generated/included.txt",
                        ],
                    )
            finally:
                context.close()

    def test_canonical_context_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            try:
                (root / "link").symlink_to(root / "Dockerfile")
            except (NotImplementedError, OSError):
                self.skipTest("symbolic links are unavailable")
            with self.assertRaisesRegex(InvalidRequestError, "symbolic links"):
                canonical_context(root)

    def test_recipe_hash_contract_is_versioned_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            context = canonical_context(root)
            try:
                self.assertRegex(context.recipe_hash, r"^sha256:[0-9a-f]{64}$")
                self.assertEqual(
                    context.recipe_hash,
                    "sha256:73eb98b3b16402cc6f14c2758a2be321366e6abfb6ea6929297fe0e9a700c399",
                )
            finally:
                context.close()


class BridgeTest(unittest.TestCase):
    def test_bridge_uses_one_persistent_background_loop(self) -> None:
        bridge = AsyncBridge()

        async def identity() -> tuple[int, int]:
            return id(asyncio.get_running_loop()), threading.get_ident()

        try:
            first = bridge.run(identity())
            second = bridge.run(identity())
        finally:
            bridge.close()
        self.assertEqual(first, second)
        self.assertNotEqual(first[1], threading.get_ident())

    def test_bridge_can_block_a_thread_which_has_a_running_loop(self) -> None:
        async def caller() -> int:
            bridge = AsyncBridge()
            try:
                return bridge.run(asyncio.sleep(0, result=42))
            finally:
                bridge.close()

        self.assertEqual(asyncio.run(caller()), 42)

    def test_sync_and_async_calls_share_the_bridge_loop(self) -> None:
        async def caller() -> None:
            bridge = AsyncBridge()

            async def identity() -> tuple[int, int]:
                return id(asyncio.get_running_loop()), threading.get_ident()

            try:
                synchronous_result = bridge.run(identity())
                asynchronous_result = await bridge.run_async(identity())
            finally:
                bridge.close()
            self.assertEqual(synchronous_result, asynchronous_result)

        asyncio.run(caller())


class SynchronousClientTest(unittest.TestCase):
    def test_resolve_image_delegates_to_async_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = Client(config(directory))
            expected = ResolvedImage(
                "image-id", "registry.example/image@sha256:digest", "sha256:digest"
            )
            client._client.resolve_image = mock.AsyncMock(return_value=expected)
            try:
                image = Image.from_registry("ubuntu:24.04")
                self.assertEqual(client.resolve_image(image, timeout=42), expected)
                client._client.resolve_image.assert_awaited_once_with(
                    image, timeout=42
                )
            finally:
                client.close()

    def test_request_delegates_to_async_client_on_bridge_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = Client(config(directory))

            async def request(method, path, body=None, query=None):
                return {
                    "method": method,
                    "path": path,
                    "body": body,
                    "query": query,
                    "thread": threading.get_ident(),
                }

            client._client._request = request  # type: ignore[method-assign]
            try:
                result = client._request("POST", "/test", {"x": 1}, {"page": 2})
            finally:
                client.close()
            self.assertEqual(result["method"], "POST")
            self.assertEqual(result["path"], "/test")
            self.assertNotEqual(result["thread"], threading.get_ident())

    def test_close_closes_async_client_and_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = Client(config(directory))
            close = mock.AsyncMock()
            client._client.close = close  # type: ignore[method-assign]
            client.close()
            client.close()
            close.assert_awaited_once_with()
            with self.assertRaisesRegex(ConnectionError, "closed"):
                client._request("GET", "/test")

    def test_list_wraps_every_async_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = Client(config(directory))
            prepare_key(client.config.paths)
            first = AsyncSandbox._from_response(
                client._client, {**SANDBOX_RESPONSE, "id": "sbx-one"}
            )
            second = AsyncSandbox._from_response(
                client._client, {**SANDBOX_RESPONSE, "id": "sbx-two"}
            )

            async def listing(*, status="active"):
                yield first
                yield second

            client._client.list_sandboxes = listing  # type: ignore[method-assign]
            try:
                self.assertEqual(
                    [sandbox.id for sandbox in client.list_sandboxes()],
                    ["sbx-one", "sbx-two"],
                )
            finally:
                client.close()


class AsyncImageTest(unittest.IsolatedAsyncioTestCase):
    async def test_registry_image_is_imported_and_returned_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(
                return_value={
                    "id": "image-id",
                    "state": "READY",
                    "managed_reference": "managed.example/image@sha256:digest",
                    "managed_digest": "sha256:digest",
                }
            )
            try:
                image = Image.from_registry(
                    "private.example/image:latest",
                    "user",
                    "secret",
                    display_name="Training image",
                )
                resolved = await client.resolve_image(image)
                self.assertEqual(resolved.id, "image-id")
                client._request.assert_awaited_once()
                request = client._request.await_args
                self.assertEqual(request.args[:2], ("POST", "/sandbox-images/from-registry"))
                self.assertEqual(
                    request.kwargs["body"],
                    {
                        "reference": "private.example/image:latest",
                        "username": "user",
                        "password": "secret",
                        "display_name": "Training image",
                    },
                )
            finally:
                await client.close()

    async def test_dockerfile_image_is_archived_uploaded_and_polled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_bytes(b"FROM scratch\n")
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(
                side_effect=[
                    {
                        "id": "image-id",
                        "state": "BUILDING",
                        "upload": {
                            "host": "203.0.113.10",
                            "port": 32022,
                            "username": "image-upload",
                            "path": "/build-context.tar",
                            "host_public_key": asyncssh.generate_private_key(
                                "ssh-ed25519"
                            ).export_public_key().decode("utf-8").strip(),
                            "expires_at": "2099-01-01T00:00:00Z",
                        },
                    },
                    {
                        "id": "image-id",
                        "state": "READY",
                        "managed_reference": "managed.example/image@sha256:digest",
                        "managed_digest": "sha256:digest",
                    },
                ]
            )
            client._upload_image_context = mock.AsyncMock()
            try:
                with mock.patch(
                    "thunder_sandbox.asynchronous.client.asyncio.sleep",
                    new=mock.AsyncMock(),
                ):
                    resolved = await client.resolve_image(
                        Image.from_dockerfile(root, display_name="Training image")
                    )
                self.assertEqual(resolved.id, "image-id")
                create_call, status_call = client._request.await_args_list
                self.assertEqual(
                    create_call.args[:2],
                    ("POST", "/sandbox-images/from-dockerfile"),
                )
                body = create_call.kwargs["body"]
                self.assertRegex(body["recipe_hash"], r"^sha256:[0-9a-f]{64}$")
                self.assertRegex(body["context_hash"], r"^sha256:[0-9a-f]{64}$")
                self.assertGreater(body["archive_bytes"], 0)
                self.assertEqual(body["display_name"], "Training image")
                upload_public_key = asyncssh.import_public_key(body["ssh_public_key"])
                archived_context = client._upload_image_context.await_args.args[0]
                upload_private_key = client._upload_image_context.await_args.kwargs[
                    "private_key"
                ]
                self.assertEqual(
                    upload_private_key.public_data, upload_public_key.public_data
                )
                self.assertEqual(
                    archived_context.archive_path.parent,
                    client.config.paths.image_build_contexts.resolve(),
                )
                self.assertFalse(archived_context.archive_path.exists())
                self.assertEqual(
                    status_call.args[:2], ("GET", "/sandbox-images/image-id")
                )
                client._upload_image_context.assert_awaited_once()
            finally:
                await client.close()

    async def test_dockerfile_context_upload_uses_pinned_sftp(self) -> None:
        class RemoteFile:
            def __init__(self) -> None:
                self.content = bytearray()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            async def write(self, data: bytes) -> None:
                self.content.extend(data)

        class SFTPClient:
            def __init__(self, remote: RemoteFile) -> None:
                self.remote = remote
                self.opened: tuple[str, str] | None = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            def open(self, path: str, mode: str) -> RemoteFile:
                self.opened = (path, mode)
                return self.remote

        class Connection:
            def __init__(self, sftp: SFTPClient) -> None:
                self.sftp = sftp
                self.closed = False

            def start_sftp_client(self) -> SFTPClient:
                return self.sftp

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "context.tar"
            archive.write_bytes(b"canonical build context")
            context = mock.Mock(
                archive_path=archive, archive_bytes=archive.stat().st_size
            )
            upload_key = asyncssh.generate_private_key("ssh-ed25519")
            host_key = asyncssh.generate_private_key("ssh-ed25519")
            remote = RemoteFile()
            sftp = SFTPClient(remote)
            connection = Connection(sftp)
            client = AsyncClient(config(directory))
            connect = mock.AsyncMock(return_value=connection)
            try:
                with mock.patch(
                    "thunder_sandbox.asynchronous.client.asyncssh.connect", connect
                ):
                    await client._upload_image_context(
                        context,
                        {
                            "host": "203.0.113.10",
                            "port": 32022,
                            "username": "image-upload",
                            "path": "/build-context.tar",
                            "host_public_key": host_key.export_public_key()
                            .decode("utf-8")
                            .strip(),
                            "expires_at": "2099-01-01T00:00:00Z",
                        },
                        private_key=upload_key,
                        deadline=None,
                    )
                self.assertEqual(remote.content, archive.read_bytes())
                self.assertEqual(sftp.opened, ("/build-context.tar", "wb"))
                self.assertTrue(connection.closed)
                call = connect.await_args
                self.assertEqual(call.args, ("203.0.113.10", 32022))
                self.assertEqual(call.kwargs["username"], "image-upload")
                self.assertEqual(call.kwargs["client_keys"], [upload_key])
                self.assertEqual(
                    call.kwargs["known_hosts"][0][0].public_data,
                    host_key.public_data,
                )
                self.assertIsNone(call.kwargs["agent_path"])
            finally:
                await client.close()

    async def test_failed_image_raises_the_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(
                return_value={
                    "id": "image-id",
                    "state": "FAILED",
                    "failure_code": "BUILD_FAILED",
                    "failure": "docker build failed",
                }
            )
            try:
                with self.assertRaisesRegex(SandboxFailedError, "docker build failed"):
                    await client.resolve_image(Image.from_registry("ubuntu:24.04"))
            finally:
                await client.close()


class AsyncSandboxTest(unittest.IsolatedAsyncioTestCase):
    def sandbox(self, directory: str) -> tuple[AsyncSandbox, AsyncClient]:
        client = AsyncClient(config(directory))
        prepare_key(client.config.paths)
        return AsyncSandbox._from_response(client, SANDBOX_RESPONSE), client

    async def test_create_waits_for_image_before_starting_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            image = Image.from_registry("ubuntu:24.04")
            resolved = ResolvedImage(
                id="a" * 64,
                managed_reference="registry.example/image@sha256:" + "a" * 64,
                managed_digest="sha256:" + "a" * 64,
            )
            events: list[str] = []

            async def resolve(
                candidate: Image, *, timeout: float | None = 7200
            ) -> ResolvedImage:
                self.assertIs(candidate, image)
                self.assertEqual(timeout, 7200)
                events.append("image-ready")
                return resolved

            async def request(
                method: str,
                path: str,
                body: object | None = None,
                **_: object,
            ) -> dict[str, object]:
                events.append(path)
                if path == "/sandboxes/start":
                    self.assertEqual(method, "POST")
                    assert isinstance(body, dict)
                    self.assertEqual(body["image_id"], resolved.id)
                    return {"id": "sbx-test"}
                return {**SANDBOX_RESPONSE, "image_id": resolved.id}

            client.resolve_image = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=resolve
            )
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=request
            )
            sandbox = await AsyncSandbox.create(image=image, client=client)
            self.assertEqual(sandbox.id, "sbx-test")
            self.assertEqual(sandbox.image_id, resolved.id)
            self.assertEqual(
                events,
                ["image-ready", "/sandboxes/start", "/sandboxes/sbx-test"],
            )
            await client.close()

    async def test_create_with_image_runs_positional_args_through_exec(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            image = Image.from_registry("ubuntu:24.04")
            resolved = ResolvedImage(
                id="a" * 64,
                managed_reference="registry.example/image@sha256:" + "a" * 64,
                managed_digest="sha256:" + "a" * 64,
            )
            client.resolve_image = mock.AsyncMock(  # type: ignore[method-assign]
                return_value=resolved
            )
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    {"id": "sbx-test"},
                    {**SANDBOX_RESPONSE, "image_id": resolved.id},
                ]
            )

            process = mock.Mock()
            with mock.patch.object(
                AsyncSandbox, "wait_until_ready", new=mock.AsyncMock()
            ) as wait_until_ready, mock.patch.object(
                AsyncSandbox, "exec", new=mock.AsyncMock(return_value=process)
            ) as exec_process:
                sandbox = await AsyncSandbox.create(
                    "echo", "hello", image=image, client=client
                )

            start_body = client._request.await_args_list[0].args[2]
            self.assertNotIn("command", start_body)
            wait_until_ready.assert_awaited_once_with(timeout=300)
            exec_process.assert_awaited_once_with("echo", "hello")
            self.assertIs(sandbox._main_process, process)
            await client.close()

    async def test_create_without_image_keeps_positional_args_as_guest_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[{"id": "sbx-test"}, SANDBOX_RESPONSE]
            )
            process = mock.Mock()
            with mock.patch.object(
                AsyncSandbox, "wait_until_ready", new=mock.AsyncMock()
            ) as wait_until_ready, mock.patch.object(
                AsyncSandbox, "exec", new=mock.AsyncMock(return_value=process)
            ) as exec_process:
                sandbox = await AsyncSandbox.create("echo", "hello", client=client)

            start_body = client._request.await_args_list[0].args[2]
            self.assertNotIn("command", start_body)
            wait_until_ready.assert_awaited_once_with(timeout=300)
            exec_process.assert_awaited_once_with("echo", "hello")
            self.assertIs(sandbox._main_process, process)
            await client.close()

    async def test_create_does_not_start_sandbox_when_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client.resolve_image = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=SandboxFailedError("docker build failed")
            )
            client._request = mock.AsyncMock()  # type: ignore[method-assign]

            with self.assertRaisesRegex(SandboxFailedError, "docker build failed"):
                await AsyncSandbox.create(
                    image=Image.from_registry("ubuntu:24.04"),
                    client=client,
                )

            client._request.assert_not_awaited()
            await client.close()

    async def test_create_without_image_does_not_resolve_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client.resolve_image = mock.AsyncMock()  # type: ignore[method-assign]
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[{"id": "sbx-test"}, SANDBOX_RESPONSE]
            )

            sandbox = await AsyncSandbox.create(client=client)

            client.resolve_image.assert_not_awaited()
            start_body = client._request.await_args_list[0].args[2]
            self.assertNotIn("image_id", start_body)
            await client.close()

    async def test_ssh_keeps_no_known_hosts_file(self) -> None:
        # The node reuses forwarded ports, so any entry outlives the sandbox
        # that wrote it and makes ssh refuse the next sandbox on that port.
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            command = sandbox.ssh.command
            self.assertIn("StrictHostKeyChecking=accept-new", command)
            self.assertIn("UserKnownHostsFile=/dev/null", command)
            self.assertNotIn("StrictHostKeyChecking=no", command)
            await client.close()

    async def test_ssh_connection_attempts_are_individually_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            credential = mock.Mock(key=mock.Mock(), certificate=mock.Mock())
            connection = mock.Mock()
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.connect",
                new=mock.AsyncMock(return_value=connection),
            ) as connect:
                self.assertIs(
                    await sandbox._open(sandbox.ssh, credential, None), connection
                )

            self.assertEqual(
                connect.await_args.kwargs["connect_timeout"],
                SSH_CONNECT_TIMEOUT_SECONDS,
            )
            self.assertEqual(connect.await_args.kwargs["keepalive_interval"], 15)
            self.assertEqual(connect.await_args.kwargs["keepalive_count_max"], 4)
            await client.close()

    async def test_api_host_key_is_pinned_on_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            prepare_key(client.config.paths)
            host_key = asyncssh.generate_private_key("ssh-ed25519").export_public_key()
            host_key_text = host_key.decode("ascii").strip()
            response = {
                **SANDBOX_RESPONSE,
                "ssh": {
                    **SANDBOX_RESPONSE["ssh"],
                    "host_key": host_key_text,
                },
            }
            sandbox = AsyncSandbox._from_response(client, response)
            # Remembered against the sandbox, in memory only: a file keyed by
            # host and port would reject the next sandbox on a reused port.
            pinned = _pinned_host_key(sandbox.id)
            self.assertIsNotNone(pinned)
            self.assertEqual(
                pinned.export_public_key().decode("ascii").strip(), host_key_text
            )
            self.assertFalse(
                [entry for entry in Path(directory).rglob("known_hosts*")],
                "pinning a host key must not create a known-hosts file",
            )
            await client.close()

    async def test_exec_uses_asyncssh_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            stdout = mock.Mock()
            stdout.read = mock.AsyncMock(return_value=b"")
            stderr = mock.Mock()
            stderr.read = mock.AsyncMock(return_value=b"")
            process = mock.Mock(
                stdin=mock.Mock(), stdout=stdout, stderr=stderr, returncode=0
            )
            process.wait_closed = mock.AsyncMock()
            connection = mock.Mock()
            connection.create_process = mock.AsyncMock(return_value=process)
            with mock.patch.object(
                sandbox, "_connect", new=mock.AsyncMock(return_value=connection)
            ):
                remote = await sandbox.exec("echo", "hello", text=False, pty=True)
                self.assertFalse(remote.is_durable)
                self.assertEqual(await remote.wait(), 0)
            connection.create_process.assert_awaited_once_with(
                "echo hello", encoding=None, term_type="xterm"
            )
            await client.close()

    async def test_pty_launch_is_not_retried_after_an_ambiguous_disconnect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            connection = FakeSSHConnection()
            connection.create_process = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=SSHDisconnected("lost after PTY request")
            )
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]

            with self.assertRaisesRegex(ConnectionError, "open a sandbox SSH session"):
                await sandbox.exec("interactive-command", pty=True)

            connection.create_process.assert_awaited_once()  # type: ignore[attr-defined]
            self.assertTrue(connection.closed)
            await client.close()

    async def test_pty_rejects_durable_output_and_retention_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            connect = mock.AsyncMock()
            with mock.patch.object(sandbox, "_connect", new=connect):
                for options in (
                    {"stdout": "discard"},
                    {"stderr": "discard"},
                    {"retain": True},
                ):
                    with self.subTest(options=options), self.assertRaisesRegex(
                        InvalidRequestError, "require pty=False"
                    ):
                        await sandbox.exec("interactive-command", pty=True, **options)
            connect.assert_not_awaited()
            await client.close()

    async def test_pty_runtime_disconnect_reports_unknown_remote_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            stdout = mock.Mock(
                read=mock.AsyncMock(side_effect=SSHDisconnected("channel lost"))
            )
            stderr = mock.Mock(read=mock.AsyncMock(return_value=b""))
            process = mock.Mock(
                stdin=mock.Mock(), stdout=stdout, stderr=stderr, returncode=None
            )
            process.wait_closed = mock.AsyncMock()
            connection = FakeSSHConnection()
            connection.create_process = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=process
            )
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]

            remote = await sandbox.exec("interactive-command", text=False, pty=True)
            with self.assertRaisesRegex(ConnectionError, "remote state is unknown"):
                await remote.wait()

            self.assertIsNone(remote.returncode)
            await sandbox._close_connection()
            await client.close()

    async def test_non_pty_exec_returns_a_durable_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            status = JobStatus(JobState.RUNNING, pid=123)
            spec = JobSpec("8" * 32, ("python", "train.py"))
            durable = mock.Mock()
            with mock.patch.object(
                sandbox,
                "_launch_detached_job",
                new=mock.AsyncMock(return_value=(spec, status)),
            ) as launch, mock.patch.object(
                sandbox, "_durable_process", return_value=durable
            ) as process:
                result = await sandbox.exec(
                    "python",
                    "train.py",
                    workdir="/workspace",
                    env={"MODEL": "large"},
                    timeout=600,
                )

            self.assertIs(result, durable)
            launch.assert_awaited_once_with(
                ("python", "train.py"),
                workdir="/workspace",
                env={"MODEL": "large"},
                stdout="capture",
                stderr="capture",
                retain=False,
            )
            process.assert_called_once_with(
                spec, status=status, timeout=600, text=True
            )
            await client.close()

    async def test_get_process_recovers_a_durable_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            status = JobStatus(JobState.FAILED, pid=456, returncode=9)
            spec = JobSpec("9" * 32, ("false",), retain=True)
            with mock.patch.object(
                sandbox, "_read_job_status", new=mock.AsyncMock(return_value=status)
            ) as read_status, mock.patch.object(
                sandbox, "_read_job_spec", new=mock.AsyncMock(return_value=spec)
            ) as read_spec:
                process = await sandbox.get_process("9" * 32, text=False)

            self.assertEqual(process.id, "9" * 32)
            self.assertEqual(process.returncode, 9)
            self.assertEqual(await process.wait(), 9)
            read_status.assert_awaited_once_with(
                "9" * 32, deadline=mock.ANY
            )
            read_spec.assert_awaited_once_with("9" * 32, deadline=mock.ANY)
            await client.close()

    async def test_get_process_rejects_an_unsafe_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            with self.assertRaises(InvalidRequestError):
                await sandbox.get_process("../escape")
            await client.close()

    async def test_output_sftp_reconnects_and_replays_the_same_byte_offset(self) -> None:
        class RemoteFile:
            def __init__(self, outcome: bytes | BaseException) -> None:
                self.outcome = outcome
                self.reads: list[tuple[int, int]] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def read(self, size: int, offset: int) -> bytes:
                self.reads.append((size, offset))
                if isinstance(self.outcome, BaseException):
                    raise self.outcome
                return self.outcome

        class SFTPClient:
            def __init__(self, remote: RemoteFile) -> None:
                self.remote = remote

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            def open(self, _path: str, mode: str) -> RemoteFile:
                self.assert_mode = mode
                return self.remote

        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            failed_file = RemoteFile(SSHDisconnected("SFTP connection lost"))
            recovered_file = RemoteFile(b"resumed")
            first = FakeSSHConnection()
            second = FakeSSHConnection()
            first.start_sftp_client = mock.Mock(  # type: ignore[attr-defined]
                return_value=SFTPClient(failed_file)
            )
            second.start_sftp_client = mock.Mock(  # type: ignore[attr-defined]
                return_value=SFTPClient(recovered_file)
            )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]

            value = await sandbox._read_job_output(
                "d" * 32, "stdout", offset=8192, size=4096
            )

            self.assertEqual(value, b"resumed")
            self.assertEqual(failed_file.reads, [(4096, 8192)])
            self.assertEqual(recovered_file.reads, [(4096, 8192)])
            self.assertTrue(first.closed)
            await sandbox._close_connection()
            await client.close()

    async def test_job_cleanup_is_idempotent_after_a_lost_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            first = FakeSSHConnection()
            first.run = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=SSHDisconnected("cleanup acknowledgement lost")
            )
            second = FakeSSHConnection()
            second.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(returncode=0, stdout="", stderr="")
            )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]

            await sandbox._cleanup_job("e" * 32)

            first_command = first.run.await_args.args[0]  # type: ignore[attr-defined]
            second_command = second.run.await_args.args[0]  # type: ignore[attr-defined]
            self.assertEqual(first_command, second_command)
            self.assertEqual(first_command.count("/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"), 1)
            await sandbox._close_connection()
            await client.close()

    async def test_job_signal_reconnects_after_a_lost_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            terminated = JobStatus(JobState.TERMINATED, pid=4242, returncode=143)
            first = FakeSSHConnection()
            first.run = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=SSHDisconnected("signal acknowledgement lost")
            )
            second = FakeSSHConnection()
            second.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(
                    returncode=0, stdout=terminated.to_json(), stderr=""
                )
            )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]

            status = await sandbox._signal_job(
                "f" * 32, signal="TERM", pid=4242
            )

            self.assertEqual(status, terminated)
            first_command = first.run.await_args.args[0]  # type: ignore[attr-defined]
            second_command = second.run.await_args.args[0]  # type: ignore[attr-defined]
            self.assertEqual(first_command, second_command)
            self.assertIn('touch -- "$job/termination.request"', first_command)
            self.assertIn('kill -TERM -- "-$expected_pid"', first_command)
            await sandbox._close_connection()
            await client.close()

    async def test_force_kill_reconciles_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            terminated = JobStatus(JobState.TERMINATED, pid=5252, returncode=137)
            connection = FakeSSHConnection()
            connection.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(
                    returncode=0, stdout=terminated.to_json(), stderr=""
                )
            )
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]

            status = await sandbox._signal_job(
                "1" * 32, signal="KILL", pid=5252
            )

            self.assertEqual(status, terminated)
            command = connection.run.await_args.args[0]  # type: ignore[attr-defined]
            self.assertIn('kill -KILL -- "-$expected_pid"', command)
            self.assertIn('"returncode":137', command)
            self.assertIn('mv -f -- "$temporary" "$status"', command)
            syntax = subprocess.run(
                ["sh", "-n"], input=command, text=True, capture_output=True
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            await sandbox._close_connection()
            await client.close()

    async def test_image_job_signal_targets_and_checks_container_process_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            terminated = JobStatus(JobState.TERMINATED, pid=6262, returncode=137)
            connection = FakeSSHConnection()
            connection.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(
                    returncode=0, stdout=terminated.to_json(), stderr=""
                )
            )
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]

            status = await sandbox._signal_job(
                "2" * 32,
                signal="KILL",
                pid=6262,
                container="thunder-sandbox",
            )

            self.assertEqual(status, terminated)
            command = connection.run.await_args.args[0]  # type: ignore[attr-defined]
            self.assertIn("docker exec thunder-sandbox /busybox cat", command)
            self.assertIn("/tmp/thunder-sandbox/jobs/22222222222222222222222222222222/pid", command)
            self.assertIn('/busybox kill -KILL -- "-$container_pid"', command)
            self.assertIn('/busybox kill -0 -- "-$container_pid"', command)
            self.assertNotIn('kill -KILL -- "-$expected_pid"', command)
            syntax = subprocess.run(
                ["sh", "-n"], input=command, text=True, capture_output=True
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            await sandbox._close_connection()
            await client.close()

    async def test_recovered_process_wait_reconnects_after_status_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            running = JobStatus(JobState.RUNNING, pid=777)
            succeeded = JobStatus(JobState.SUCCEEDED, pid=777, returncode=0)
            spec = JobSpec("a" * 32, ("true",), retain=True)
            first = FakeSSHConnection()
            first.run = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=[
                    mock.Mock(returncode=0, stdout=spec.to_json(), stderr=""),
                    mock.Mock(returncode=0, stdout=running.to_json(), stderr=""),
                    SSHDisconnected("status channel lost"),
                ]
            )
            second = FakeSSHConnection()
            second.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(
                    returncode=0, stdout=succeeded.to_json(), stderr=""
                )
            )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]

            process = await sandbox.get_process("a" * 32)
            self.assertEqual(await process.wait(), 0)

            self.assertTrue(first.closed)
            self.assertEqual(second.run.await_count, 1)  # type: ignore[attr-defined]
            await sandbox._close_connection()
            await client.close()

    async def test_detached_launch_returns_the_remote_job_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            acknowledgement = JobStatus(JobState.RUNNING, pid=4321)
            result = mock.Mock(
                returncode=0, stdout=acknowledgement.to_json(), stderr=""
            )
            connection = mock.Mock()
            connection.is_closed.return_value = False
            connection.run = mock.AsyncMock(return_value=result)
            sandbox._ssh_manager._connection = connection
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.new_job_id",
                return_value="f" * 32,
            ):
                spec, status = await sandbox._launch_detached_job(
                    ("echo", "hello"), workdir="/work dir", env={"VALUE": "a b"}
                )

            self.assertEqual(spec.job_id, "f" * 32)
            self.assertEqual(status, acknowledgement)
            submitted = connection.run.await_args.args[0]
            self.assertIn("nohup setsid", submitted)
            self.assertIn("workdir", submitted)
            self.assertIn("echo hello", submitted)
            connection.run.assert_awaited_once_with(
                submitted, check=False, encoding="utf-8"
            )
            await client.close()

    async def test_detached_launch_targets_the_image_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            sandbox = AsyncSandbox._from_response(
                client, {**SANDBOX_RESPONSE, "image_id": "a" * 64}
            )
            result = mock.Mock(
                returncode=0,
                stdout=JobStatus(JobState.STARTING, pid=987).to_json(),
                stderr="",
            )
            connection = mock.Mock(run=mock.AsyncMock(return_value=result))
            connection.is_closed.return_value = False
            sandbox._ssh_manager._connection = connection
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.new_job_id",
                return_value="1" * 32,
            ):
                await sandbox._launch_detached_job(
                    ("python", "train.py"), workdir="/workspace", env=None
                )

            submitted = connection.run.await_args.args[0]
            self.assertIn(
                "sudo --non-interactive docker exec --interactive", submitted
            )
            self.assertIn("thunder-sandbox /busybox setsid /busybox sh", submitted)
            self.assertIn("/tmp/thunder-sandbox/jobs/11111111111111111111111111111111", submitted)
            self.assertIn("python train.py", submitted)
            await client.close()

    async def test_detached_launch_reconciles_a_lost_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            acknowledgement = JobStatus(JobState.RUNNING, pid=2468)
            first = FakeSSHConnection()
            first.run = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=SSHDisconnected("lost after remote launch")
            )
            second = FakeSSHConnection()
            second.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(
                    returncode=0, stdout=acknowledgement.to_json(), stderr=""
                )
            )
            opened = mock.AsyncMock(side_effect=[first, second])
            sandbox._ssh_manager = SSHConnectionManager(
                opened,  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.new_job_id",
                return_value="3" * 32,
            ):
                spec, status = await sandbox._launch_detached_job(
                    ("touch", "/tmp/exactly-once"), workdir=None, env=None
                )

            self.assertEqual(spec.job_id, "3" * 32)
            self.assertEqual(status, acknowledgement)
            self.assertTrue(first.closed)
            self.assertEqual(opened.await_count, 2)
            first_submission = first.run.await_args.args[0]  # type: ignore[attr-defined]
            second_submission = second.run.await_args.args[0]  # type: ignore[attr-defined]
            for submitted in (first_submission, second_submission):
                self.assertIn("/33333333333333333333333333333333", submitted)
                self.assertIn("touch", submitted)
                self.assertIn("/tmp/exactly-once", submitted)
            self.assertNotEqual(first_submission, second_submission)
            await sandbox._close_connection()
            await client.close()

    async def test_connection_renews_a_rejected_certificate_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            rejected = mock.Mock()
            renewed = mock.Mock()
            connection = FakeSSHConnection()
            connection.get_server_host_key = mock.Mock(return_value="ssh-ed25519 AAAA")  # type: ignore[attr-defined]
            client._credentials.ensure = mock.AsyncMock(return_value=rejected)  # type: ignore[method-assign]
            client._credentials.renew = mock.AsyncMock(return_value=renewed)  # type: ignore[method-assign]
            sandbox._open = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[asyncssh.PermissionDenied("rejected"), connection]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox._pinned_host_key",
                return_value=mock.Mock(),
            ):
                self.assertIs(await sandbox._open_connection(), connection)

            client._credentials.renew.assert_awaited_once_with(  # type: ignore[attr-defined]
                client, rejected=rejected
            )
            self.assertEqual(sandbox._open.await_count, 2)  # type: ignore[attr-defined]
            await client.close()

    async def test_exec_enters_image_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            sandbox = AsyncSandbox._from_response(
                client, {**SANDBOX_RESPONSE, "image_id": "a" * 64}
            )
            stdout = mock.Mock(read=mock.AsyncMock(return_value=b""))
            stderr = mock.Mock(read=mock.AsyncMock(return_value=b""))
            process = mock.Mock(
                stdin=mock.Mock(), stdout=stdout, stderr=stderr, returncode=0
            )
            process.wait_closed = mock.AsyncMock()
            connection = mock.Mock()
            connection.create_process = mock.AsyncMock(return_value=process)
            with mock.patch.object(
                sandbox, "_connect", new=mock.AsyncMock(return_value=connection)
            ):
                remote = await sandbox.exec(
                    "echo",
                    "hello",
                    workdir="/work dir",
                    env={"VALUE": "a b", "REMOVE": None},
                    text=False,
                    pty=True,
                )
                self.assertEqual(await remote.wait(), 0)
            connection.create_process.assert_awaited_once_with(
                "sudo --non-interactive docker exec --interactive --tty "
                "--workdir '/work dir' --env 'VALUE=a b' thunder-sandbox "
                "/busybox env -u REMOVE echo hello",
                encoding=None,
                term_type="xterm",
            )
            await client.close()

    async def test_upload_restarts_the_complete_staged_transfer_after_disconnect(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            source = Path(directory) / "model.bin"
            source.write_bytes(b"model")
            first = FakeSSHConnection()
            second = FakeSSHConnection()
            for connection in (first, second):
                connection.run = mock.AsyncMock(  # type: ignore[attr-defined]
                    return_value=mock.Mock(returncode=0, stdout="", stderr="")
                )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]
            transfer_attempts = 0

            async def disconnected_transfer(*_args: object, **_kwargs: object) -> None:
                nonlocal transfer_attempts
                transfer_attempts += 1
                if transfer_attempts == 1:
                    first.close()
                    raise SSHDisconnected("transfer lost")

            with mock.patch.object(
                sandbox,
                "_resolve_remote_upload_target",
                new=mock.AsyncMock(return_value="/workspace/model.bin"),
            ), mock.patch.object(
                sandbox, "_publish_remote_transfer", new=mock.AsyncMock()
            ) as publish, mock.patch.object(
                sandbox, "_cleanup_remote_transfer_paths", new=mock.AsyncMock()
            ) as cleanup, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.scp",
                new=mock.AsyncMock(side_effect=disconnected_transfer),
            ) as scp, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.uuid.uuid4",
                return_value=mock.Mock(hex="retry"),
            ):
                await sandbox.upload(source, "/workspace/model.bin")

            self.assertEqual(scp.await_count, 2)
            first_stage = scp.await_args_list[0].args[1][1]
            second_stage = scp.await_args_list[1].args[1][1]
            self.assertEqual(first_stage, second_stage)
            self.assertIn(".thunder-transfer-retry.stage", first_stage)
            self.assertTrue(first.closed)
            publish.assert_awaited_once_with(
                first_stage,
                "/workspace/model.bin",
                "/workspace/model.bin.thunder-transfer-retry.backup",
            )
            cleanup.assert_awaited_once()
            await sandbox._close_connection()
            await client.close()

    async def test_download_keeps_destination_intact_until_retry_succeeds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            destination = Path(directory) / "result.bin"
            destination.write_bytes(b"old-result")
            first = FakeSSHConnection()
            second = FakeSSHConnection()
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]
            observed_destinations: list[bytes] = []

            async def transfer(_source: object, staged: str, **_kwargs: object) -> None:
                observed_destinations.append(destination.read_bytes())
                Path(staged).write_bytes(
                    b"partial" if len(observed_destinations) == 1 else b"complete"
                )
                if len(observed_destinations) == 1:
                    first.close()
                    raise SSHDisconnected("download lost")

            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.scp",
                new=mock.AsyncMock(side_effect=transfer),
            ):
                await sandbox.download("/workspace/result.bin", destination)

            self.assertEqual(observed_destinations, [b"old-result", b"old-result"])
            self.assertEqual(destination.read_bytes(), b"complete")
            self.assertTrue(first.closed)
            self.assertEqual(
                list(Path(directory).glob(".result.bin.thunder-transfer-*")), []
            )
            await sandbox._close_connection()
            await client.close()

    async def test_directory_contents_download_preserves_existing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            destination = Path(directory) / "results"
            destination.mkdir()
            (destination / "existing.txt").write_text("keep", encoding="utf-8")
            connection = FakeSSHConnection()
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]

            async def transfer(_source: object, staged: str, **_kwargs: object) -> None:
                stage = Path(staged)
                stage.mkdir()
                (stage / "new.txt").write_text("new", encoding="utf-8")

            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.scp",
                new=mock.AsyncMock(side_effect=transfer),
            ):
                await sandbox.download(
                    "/workspace/results/.", destination, recursive=True
                )

            self.assertEqual(
                (destination / "existing.txt").read_text(encoding="utf-8"), "keep"
            )
            self.assertEqual(
                (destination / "new.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertEqual(
                list(Path(directory).glob(".results.thunder-*")), []
            )
            await sandbox._close_connection()
            await client.close()

    async def test_upload_publication_retries_after_lost_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            first = FakeSSHConnection()
            first.run = mock.AsyncMock(  # type: ignore[attr-defined]
                side_effect=SSHDisconnected("publish acknowledgement lost")
            )
            second = FakeSSHConnection()
            second.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(returncode=0, stdout="", stderr="")
            )
            sandbox._ssh_manager = SSHConnectionManager(
                mock.AsyncMock(return_value=second),  # type: ignore[arg-type]
                sleep=mock.AsyncMock(),
                jitter=lambda _start, _end: 0.0,
            )
            sandbox._ssh_manager._connection = first  # type: ignore[assignment]

            await sandbox._publish_remote_transfer(
                "/workspace/.stage", "/workspace/final", "/workspace/.backup"
            )

            first_command = first.run.await_args.args[0]  # type: ignore[attr-defined]
            second_command = second.run.await_args.args[0]  # type: ignore[attr-defined]
            self.assertEqual(first_command, second_command)
            self.assertIn('mv -T -f -- "$stage" "$target"', first_command)
            syntax = subprocess.run(
                ["sh", "-n"], input=first_command, text=True, capture_output=True
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            await sandbox._close_connection()
            await client.close()

    async def test_upload_to_image_container_uses_isolated_guest_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            sandbox = AsyncSandbox._from_response(
                client, {**SANDBOX_RESPONSE, "image_id": "a" * 64}
            )
            source = Path(directory) / "payload.txt"
            source.write_text("payload", encoding="utf-8")
            connection = FakeSSHConnection()
            connection.run = mock.AsyncMock(  # type: ignore[attr-defined]
                return_value=mock.Mock(returncode=0, stdout="", stderr="")
            )
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]
            with mock.patch.object(
                sandbox, "_run_guest_command", new=mock.AsyncMock()
            ) as guest, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.scp",
                new=mock.AsyncMock(),
            ) as scp, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.uuid.uuid4",
                return_value=mock.Mock(hex="transfer"),
            ):
                await sandbox.upload(source, "/workspace/payload.txt")

            stage = "/tmp/thunder-sandbox-transfer-transfer"
            scp.assert_awaited_once_with(
                str(source), (connection, stage + "/"), recurse=False
            )
            self.assertEqual(
                [call.args for call in guest.await_args_list],
                [
                    (
                        "sudo",
                        "--non-interactive",
                        "docker",
                        "cp",
                        stage + "/payload.txt",
                        "thunder-sandbox:/workspace/payload.txt",
                    ),
                ],
            )
            await client.close()

    async def test_download_from_image_container_uses_isolated_guest_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            sandbox = AsyncSandbox._from_response(
                client, {**SANDBOX_RESPONSE, "image_id": "a" * 64}
            )
            destination = Path(directory) / "result.txt"
            connection = FakeSSHConnection()
            sandbox._ssh_manager._connection = connection  # type: ignore[assignment]
            with mock.patch.object(
                sandbox, "_run_guest_command", new=mock.AsyncMock()
            ) as guest, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncssh.scp",
                new=mock.AsyncMock(),
            ) as scp, mock.patch(
                "thunder_sandbox.asynchronous.sandbox._publish_local_transfer"
            ), mock.patch(
                "thunder_sandbox.asynchronous.sandbox.uuid.uuid4",
                return_value=mock.Mock(hex="transfer"),
            ):
                await sandbox.download("/workspace/result.txt", destination)

            stage = "/tmp/thunder-sandbox-transfer-transfer"
            scp_source, scp_destination = scp.await_args.args
            self.assertEqual(scp_source, (connection, stage + "/result.txt"))
            self.assertEqual(Path(scp_destination).name, "result.txt")
            self.assertNotEqual(scp_destination, str(destination))
            self.assertEqual(scp.await_args.kwargs, {"recurse": False})
            self.assertEqual(
                [call.args for call in guest.await_args_list],
                [
                    ("mkdir", "--", stage),
                    (
                        "sudo",
                        "--non-interactive",
                        "docker",
                        "cp",
                        "thunder-sandbox:/workspace/result.txt",
                        stage + "/",
                    ),
                ],
            )
            await client.close()

    async def test_wait_until_ready_holds_a_server_side_wait_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[still_starting(), SANDBOX_RESPONSE]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep:
                self.assertIs(await sandbox.wait_until_ready(), sandbox)
            # A closed window is the wait's normal answer, so the next one
            # opens at once: a pause is where readiness could go unnoticed.
            sleep.assert_not_awaited()
            self.assertEqual(client._request.await_count, 2)
            for call in client._request.await_args_list:
                self.assertEqual(call.args, ("GET", "/sandboxes/sbx-test/wait"))
                window = call.kwargs["query"]["timeout_seconds"]
                self.assertGreater(window, 0)
                self.assertLessEqual(window, WAIT_WINDOW_MAX_SECONDS)
                # The client decides when to give up on a request, and only
                # after the server has had its whole window to answer.
                self.assertGreater(call.kwargs["timeout"], window)
            await client.close()

    async def test_wait_window_is_bounded_by_the_client_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[SANDBOX_RESPONSE]
            )
            await sandbox.wait_until_ready(timeout=5)
            window = client._request.await_args.kwargs["query"]["timeout_seconds"]
            self.assertGreater(window, 0)
            self.assertLessEqual(window, 5)
            await client.close()

    async def test_wait_until_ready_times_out_on_the_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)

            async def hold_then_close_window(*args: object, **kwargs: object) -> None:
                await asyncio.sleep(0.02)
                raise still_starting()

            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=hold_then_close_window
            )
            with self.assertRaisesRegex(SandboxTimeoutError, "did not become ready"):
                await sandbox.wait_until_ready(timeout=0.1)
            self.assertGreaterEqual(client._request.await_count, 1)
            # Bounded by the deadline, not by an endless stream of windows.
            self.assertLess(client._request.await_count, 20)
            await client.close()

    async def test_wait_until_ready_falls_back_to_polling_without_the_endpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    route_missing(),
                    {**SANDBOX_RESPONSE, "status": "created", "ssh": None},
                    SANDBOX_RESPONSE,
                ]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep:
                self.assertIs(await sandbox.wait_until_ready(), sandbox)
            sleep.assert_awaited_once()
            self.assertEqual(
                [call.args for call in client._request.await_args_list],
                [
                    ("GET", "/sandboxes/sbx-test/wait"),
                    ("GET", "/sandboxes/sbx-test"),
                    ("GET", "/sandboxes/sbx-test"),
                ],
            )
            # Remembered per client: the next wait polls from the start.
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[SANDBOX_RESPONSE]
            )
            await sandbox.wait_until_ready()
            client._request.assert_awaited_once_with("GET", "/sandboxes/sbx-test")
            await client.close()

    async def test_wait_until_ready_reports_a_missing_sandbox(self) -> None:
        # The endpoint's 404 looks the same for a missing route and a missing
        # sandbox; a plain read tells them apart and is the error to surface.
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[route_missing(), route_missing()]
            )
            with self.assertRaises(NotFoundError):
                await sandbox.wait_until_ready()
            self.assertEqual(client._request.await_count, 2)
            self.assertTrue(client._wait_endpoint_available)
            await client.close()

    async def test_wait_until_ready_fails_on_a_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[{**SANDBOX_RESPONSE, "status": "failed", "ssh": None}]
            )
            with self.assertRaisesRegex(SandboxFailedError, "status: failed"):
                await sandbox.wait_until_ready()
            await client.close()

    async def test_terminate_waits_for_startup_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            sandbox = AsyncSandbox._from_response(
                client, {**SANDBOX_RESPONSE, "status": "created", "ssh": None}
            )
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    still_starting(),
                    SANDBOX_RESPONSE,
                    {},
                    {**SANDBOX_RESPONSE, "status": "finished"},
                ]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep:
                await sandbox.terminate()
            sleep.assert_not_awaited()
            self.assertEqual(
                [call.args for call in client._request.await_args_list],
                [
                    ("GET", "/sandboxes/sbx-test/wait"),
                    ("GET", "/sandboxes/sbx-test/wait"),
                    ("POST", "/sandboxes/sbx-test/stop"),
                    ("GET", "/sandboxes/sbx-test"),
                ],
            )
            self.assertEqual(sandbox.status, SandboxStatus.FINISHED)
            await client.close()

    async def test_wait_survives_retryable_polling_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    ServiceUnavailableError("retry", retry_after=3),
                    SANDBOX_RESPONSE,
                ]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep, mock.patch(
                "thunder_sandbox.asynchronous.sandbox.random.uniform", return_value=0
            ):
                self.assertIs(await sandbox.wait_until_ready(), sandbox)
            sleep.assert_awaited_once_with(3)
            await client.close()

    async def test_create_rolls_back_after_allocation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            prepare_key(client.config.paths)
            # The sandbox starts, then becoming usable fails: the half-created
            # sandbox must be stopped rather than left running and billing.
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[{"id": "sbx-test"}, SandboxFailedError("boom"), {}]
            )
            with self.assertRaises(SandboxFailedError):
                await AsyncSandbox.create(client=client)
            self.assertEqual(
                [call.args[:2] for call in client._request.await_args_list],
                [
                    ("POST", "/sandboxes/start"),
                    ("GET", "/sandboxes/sbx-test"),
                    ("POST", "/sandboxes/sbx-test/stop"),
                ],
            )
            await client.close()

    async def test_implicit_client_is_closed_by_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[SANDBOX_RESPONSE, {}, {**SANDBOX_RESPONSE, "status": "finished"}]
            )
            client.close = mock.AsyncMock()  # type: ignore[method-assign]
            with mock.patch.object(AsyncClient, "from_cli", return_value=client):
                sandbox = await AsyncSandbox.from_id("sbx-test")
            await sandbox.terminate()
            client.close.assert_awaited_once_with()

    async def test_unknown_status_and_gpu_are_forward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            response = {
                **SANDBOX_RESPONSE,
                "status": "new-state",
                "spec": {**SANDBOX_RESPONSE["spec"], "gpu_type": "L40S"},
            }
            sandbox = AsyncSandbox._from_response(client, response)
            self.assertEqual(sandbox.status, SandboxStatus.UNKNOWN)
            self.assertEqual(sandbox.info.resources.gpu_type, GPUType.UNKNOWN)
            await client.close()

    async def test_update_network_policy_replaces_the_complete_policy(self) -> None:
        cases = [
            ("open", {}, "open", [], [], [], []),
            ("closed", {"block_network": True}, "closed", [], [], [], []),
            (
                "CIDR restricted",
                {"outbound_cidr_allowlist": ["203.0.113.7/24"]},
                "restricted",
                ["203.0.113.7/24"],
                ["*"],
                ["203.0.113.0/24"],
                ["*"],
            ),
            (
                "domain restricted",
                {"outbound_domain_allowlist": ["PACKAGES.EXAMPLE.COM"]},
                "restricted",
                ["0.0.0.0/0"],
                ["PACKAGES.EXAMPLE.COM"],
                ["0.0.0.0/0"],
                ["packages.example.com"],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            try:
                for (
                    name,
                    options,
                    access,
                    cidrs,
                    domains,
                    accepted_cidrs,
                    accepted_domains,
                ) in cases:
                    with self.subTest(name=name):
                        client._request = mock.AsyncMock(  # type: ignore[method-assign]
                            return_value={
                                "id": sandbox.id,
                                "network_policy": {
                                    "internet_access": access,
                                    "cidr_allowlist": accepted_cidrs,
                                    "domain_allowlist": accepted_domains,
                                },
                            }
                        )
                        await sandbox.update_network_policy(**options)
                        client._request.assert_awaited_once_with(
                            "PATCH",
                            "/sandboxes/sbx-test/network-policy",
                            {
                                "network_policy": {
                                    "internet_access": access,
                                    "cidr_allowlist": cidrs,
                                    "domain_allowlist": domains,
                                }
                            },
                        )
                        self.assertEqual(
                            sandbox.info.network_policy.internet_access, access
                        )
                        self.assertEqual(
                            sandbox.info.network_policy.outbound_cidr_allowlist,
                            tuple(accepted_cidrs),
                        )
                        self.assertEqual(
                            sandbox.info.network_policy.outbound_domain_allowlist,
                            tuple(accepted_domains),
                        )
            finally:
                await client.close()

    async def test_update_network_policy_rejects_closed_with_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock()  # type: ignore[method-assign]
            try:
                with self.assertRaises(InvalidRequestError):
                    await sandbox.update_network_policy(
                        block_network=True,
                        outbound_cidr_allowlist=[],
                    )
                client._request.assert_not_awaited()
            finally:
                await client.close()


class SynchronousSandboxTest(unittest.TestCase):
    def sandbox(self, directory: str) -> tuple[Sandbox, Client, AsyncSandbox]:
        client = Client(config(directory))
        prepare_key(client.config.paths)
        asynchronous = AsyncSandbox._from_response(client._client, SANDBOX_RESPONSE)
        return Sandbox._from_async(client, asynchronous), client, asynchronous

    def test_properties_are_direct_views_of_async_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)
            try:
                self.assertEqual(sandbox.id, asynchronous.id)
                self.assertEqual(sandbox.name, "worker")
                self.assertEqual(sandbox.status, SandboxStatus.READY)
                self.assertEqual(sandbox.ssh.port, 2222)
            finally:
                client.close()

    def test_lifecycle_methods_block_on_async_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)
            asynchronous.refresh = mock.AsyncMock(return_value=asynchronous)  # type: ignore[method-assign]
            asynchronous.poll = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]
            asynchronous.wait = mock.AsyncMock(return_value=0)  # type: ignore[method-assign]
            asynchronous.wait_until_ready = mock.AsyncMock(return_value=asynchronous)  # type: ignore[method-assign]
            asynchronous.update_network_policy = mock.AsyncMock()  # type: ignore[method-assign]
            asynchronous.terminate = mock.AsyncMock()  # type: ignore[method-assign]
            try:
                self.assertIs(sandbox.refresh(), sandbox)
                self.assertIsNone(sandbox.poll())
                self.assertEqual(sandbox.wait(timeout=3), 0)
                self.assertIs(sandbox.wait_until_ready(timeout=4), sandbox)
                sandbox.update_network_policy(block_network=True)
                sandbox.terminate(timeout=5)
            finally:
                client.close()
            asynchronous.refresh.assert_awaited_once_with()  # type: ignore[attr-defined]
            asynchronous.wait.assert_awaited_once_with(timeout=3)  # type: ignore[attr-defined]
            asynchronous.update_network_policy.assert_awaited_once_with(  # type: ignore[attr-defined]
                block_network=True,
                outbound_cidr_allowlist=None,
                outbound_domain_allowlist=None,
            )
            asynchronous.terminate.assert_awaited_once_with(timeout=5)  # type: ignore[attr-defined]

    def test_exec_and_streams_remain_on_persistent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)

            class Reader:
                def __init__(self):
                    self.value = "output"

                async def read(self, n=-1):
                    value, self.value = self.value, ""
                    return value

                async def readline(self):
                    return ""

            class Writer:
                def __init__(self):
                    self.values = []

                def write(self, value):
                    self.values.append(value)

                async def drain(self):
                    pass

                def write_eof(self):
                    pass

            raw = mock.Mock(stdin=Writer(), stdout=Reader(), stderr=Reader(), returncode=0)
            raw.wait_closed = mock.AsyncMock()
            remote = AsyncProcess(raw)
            asynchronous.exec = mock.AsyncMock(return_value=remote)  # type: ignore[method-assign]
            try:
                process = sandbox.exec("echo", "hello")
                self.assertFalse(process.is_durable)
                self.assertEqual(process.stdin.write("input"), 5)
                self.assertEqual(process.stdout.read(), "output")
                self.assertEqual(process.wait(), 0)
            finally:
                client.close()

    def test_get_process_wraps_a_recovered_native_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)
            raw = mock.Mock(
                id="a" * 32,
                stdin=mock.Mock(),
                stdout=mock.Mock(),
                stderr=mock.Mock(),
                returncode=0,
            )
            asynchronous.get_process = mock.AsyncMock(return_value=raw)  # type: ignore[method-assign]
            try:
                process = sandbox.get_process("a" * 32, text=False)
                self.assertEqual(process.id, "a" * 32)
                self.assertEqual(process.returncode, 0)
            finally:
                client.close()
            asynchronous.get_process.assert_awaited_once_with(  # type: ignore[attr-defined]
                "a" * 32, text=False
            )

    def test_wait_does_not_consume_process_output(self) -> None:
        async def exercise() -> None:
            class Reader:
                def __init__(self, value: str) -> None:
                    self.value = value

                async def read(self, n=-1):
                    value, self.value = self.value, ""
                    return value

            raw = mock.Mock(
                stdin=mock.Mock(),
                stdout=Reader("stdout"),
                stderr=Reader("stderr"),
                returncode=0,
            )
            raw.wait = mock.AsyncMock()
            raw.wait_closed = mock.AsyncMock()
            process = AsyncProcess(raw)

            self.assertEqual(await process.wait(), 0)
            self.assertEqual(await process.stdout.read(), "stdout")
            self.assertEqual(await process.stderr.read(), "stderr")
            raw.wait.assert_not_awaited()
            raw.wait_closed.assert_awaited_once_with()

        asyncio.run(exercise())

    def test_process_stdin_is_transport_neutral(self) -> None:
        async def exercise() -> None:
            raw_stdin = mock.Mock()
            raw_stdin.drain = mock.AsyncMock()
            stdout = mock.Mock()
            stdout.read = mock.AsyncMock(return_value="")
            stderr = mock.Mock()
            stderr.read = mock.AsyncMock(return_value="")
            raw = mock.Mock(
                stdin=raw_stdin,
                stdout=stdout,
                stderr=stderr,
                returncode=0,
            )
            process = AsyncProcess(raw)

            self.assertNotIsInstance(process.stdin, asyncssh.SSHWriter)
            self.assertEqual(process.stdin.write("answer\n"), 7)
            await process.stdin.drain()
            process.stdin.write_eof()
            raw_stdin.write.assert_called_once_with("answer\n")
            raw_stdin.drain.assert_awaited_once_with()
            raw_stdin.write_eof.assert_called_once_with()

        asyncio.run(exercise())

    def test_wait_drains_large_output_without_losing_it(self) -> None:
        async def exercise() -> None:
            expected = "x" * (3 * 1024 * 1024)

            class Reader:
                def __init__(self, value: str) -> None:
                    self.value = value

                async def read(self, n=-1):
                    if not self.value:
                        return ""
                    value, self.value = self.value[:65536], self.value[65536:]
                    await asyncio.sleep(0)
                    return value

            raw = mock.Mock(
                stdin=mock.Mock(),
                stdout=Reader(expected),
                stderr=Reader(""),
                returncode=0,
            )
            raw.wait_closed = mock.AsyncMock()
            process = AsyncProcess(raw)
            self.assertEqual(await process.wait(), 0)
            self.assertEqual(await process.stdout.read(), expected)

        asyncio.run(exercise())

    def test_async_timeout_propagates_through_sync_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)
            asynchronous.wait = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=SandboxTimeoutError("timed out")
            )
            try:
                with self.assertRaises(SandboxTimeoutError):
                    sandbox.wait(timeout=1)
            finally:
                client.close()

    def test_async_named_lifecycle_methods_use_same_public_object(self) -> None:
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as directory:
                sandbox, client, asynchronous = self.sandbox(directory)
                asynchronous.refresh = mock.AsyncMock(return_value=asynchronous)  # type: ignore[method-assign]
                asynchronous.poll = mock.AsyncMock(return_value=None)  # type: ignore[method-assign]
                asynchronous.wait = mock.AsyncMock(return_value=0)  # type: ignore[method-assign]
                asynchronous.wait_until_ready = mock.AsyncMock(return_value=asynchronous)  # type: ignore[method-assign]
                asynchronous.update_network_policy = mock.AsyncMock()  # type: ignore[method-assign]
                try:
                    self.assertIs(await sandbox.refresh_async(), sandbox)
                    self.assertIsNone(await sandbox.poll_async())
                    self.assertEqual(await sandbox.wait_async(timeout=3), 0)
                    self.assertIs(
                        await sandbox.wait_until_ready_async(timeout=4), sandbox
                    )
                    await sandbox.update_network_policy_async(
                        outbound_domain_allowlist=["example.com"]
                    )
                finally:
                    await client.close_async()
                asynchronous.update_network_policy.assert_awaited_once_with(  # type: ignore[attr-defined]
                    block_network=False,
                    outbound_cidr_allowlist=None,
                    outbound_domain_allowlist=["example.com"],
                )

        asyncio.run(exercise())


class AsyncClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_timeout_bounds_one_request_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            seen: list[dict[str, object]] = []

            class Response:
                status = 200
                headers: dict[str, str] = {}

                async def read(self) -> bytes:
                    return b"{}"

                async def __aenter__(self) -> "Response":
                    return self

                async def __aexit__(self, *exc: object) -> None:
                    return None

            def request(method: str, url: str, **kwargs: object) -> Response:
                seen.append(kwargs)
                return Response()

            session = mock.Mock()
            session.request = request
            client._get_session = lambda: session  # type: ignore[method-assign]
            try:
                await client._request("GET", "/a")
                await client._request("GET", "/b", timeout=45)
                await client._request("GET", "/c")
            finally:
                await client.close()
            self.assertNotIn("timeout", seen[0])
            self.assertEqual(seen[1]["timeout"].total, 45)  # type: ignore[union-attr]
            self.assertNotIn("timeout", seen[2])

    def test_closed_wait_window_maps_to_its_own_error(self) -> None:
        error = _api_error(
            408, "sandbox_wait_timeout", "still starting", {"Retry-After": "0"}
        )
        self.assertIsInstance(error, _WaitWindowElapsedError)
        self.assertEqual(error.retry_after, 0)
        self.assertNotIn("_WaitWindowElapsedError", thunder.__dict__)


class ErrorContractTest(unittest.TestCase):
    def test_error_hierarchy_is_shared(self) -> None:
        self.assertTrue(issubclass(CapacityError, Exception))
        self.assertTrue(issubclass(RateLimitError, Exception))
        error = CapacityError(
            "unavailable", code="sandbox_capacity_unavailable", status=503,
            retry_after=20,
        )
        self.assertEqual(error.retry_after, 20)

    def test_closed_bridge_rejects_new_work_without_leaking_coroutine(self) -> None:
        bridge = AsyncBridge()
        bridge.close()
        with self.assertRaises(RuntimeError):
            bridge.run(asyncio.sleep(0))



class CredentialTest(unittest.IsolatedAsyncioTestCase):
    """One key per machine, one certificate per organization, renewed in time."""

    def _client(self, directory: str, expires_in: float = 12 * 3600) -> AsyncClient:
        client = AsyncClient(config(directory))
        ca = asyncssh.generate_private_key("ssh-ed25519")

        async def issue(method, path, body=None, query=None):
            signed = ca.generate_user_certificate(
                asyncssh.import_public_key(body["ssh_public_key"]),
                "thunder", principals=["thunder-org-org-1"],
                valid_before=int(time.time() + expires_in),
            )
            return {
                "ssh_certificate": signed.export_certificate().decode("ascii").strip(),
                "expires_at": datetime.fromtimestamp(
                    time.time() + expires_in, timezone.utc
                ).isoformat(),
            }

        client._request = mock.AsyncMock(side_effect=issue)  # type: ignore[method-assign]
        return client

    async def test_a_certificate_is_minted_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            credential = await client._credentials.ensure(client)
            self.assertTrue(credential.is_usable())
            paths = client.config.paths
            self.assertTrue(paths.ssh_key.is_file())
            self.assertTrue(paths.ssh_certificate.is_file())
            await client.close()

    async def test_a_usable_certificate_is_not_reminted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            await client._credentials.ensure(client)
            await client._credentials.ensure(client)
            self.assertEqual(client._request.await_count, 1)
            await client.close()

    async def test_a_cached_credential_survives_a_new_client(self) -> None:
        # A second process must reuse the cached credential rather than mint
        # one, which is the whole point of writing it down.
        with tempfile.TemporaryDirectory() as directory:
            first = self._client(directory)
            await first._credentials.ensure(first)
            await first.close()
            second = self._client(directory)
            await second._credentials.ensure(second)
            second._request.assert_not_awaited()
            await second.close()

    async def test_an_expiring_certificate_is_renewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # Inside the renewal margin, so it must not be handed out.
            client = self._client(directory, expires_in=60)
            await client._credentials.ensure(client)
            await client._credentials.ensure(client)
            self.assertEqual(client._request.await_count, 2)
            await client.close()

    async def test_the_key_is_kept_when_only_the_certificate_expires(self) -> None:
        # Certificates already issued name this key, so regenerating it would
        # strand them.
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            first = await client._credentials.ensure(client)
            original = client.config.paths.ssh_key.read_bytes()
            client._credentials._current = None
            client.config.paths.ssh_certificate_meta.unlink()
            second = await client._credentials.ensure(client)
            self.assertEqual(client.config.paths.ssh_key.read_bytes(), original)
            self.assertEqual(
                first.key.export_public_key(), second.key.export_public_key()
            )
            await client.close()

    async def test_an_unwritable_cache_still_yields_a_credential(self) -> None:
        # A read-only home must not stop a client connecting: nothing about
        # the credential requires it to be written down.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "home"
            root.mkdir()
            client = self._client(str(root))
            with mock.patch.object(
                Path, "mkdir", side_effect=PermissionError("read-only cache")
            ):
                credential = await client._credentials.ensure(client)
            self.assertTrue(credential.is_usable())
            self.assertFalse(client.config.paths.ssh_key.exists())
            await client.close()

    async def test_a_refused_certificate_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(return_value={})  # type: ignore[method-assign]
            with self.assertRaises(SandboxError):
                await client._credentials.ensure(client)
            await client.close()

    async def test_concurrent_rejections_share_one_certificate_renewal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            rejected = await client._credentials.ensure(client)
            replacement = mock.Mock()
            replacement.is_usable.return_value = True
            client._credentials._mint = mock.AsyncMock(return_value=replacement)  # type: ignore[method-assign]

            first, second = await asyncio.gather(
                client._credentials.renew(client, rejected=rejected),
                client._credentials.renew(client, rejected=rejected),
            )

            self.assertIs(first, replacement)
            self.assertIs(second, replacement)
            client._credentials._mint.assert_awaited_once_with(  # type: ignore[attr-defined]
                client, reuse=mock.ANY, replace=True
            )
            await client.close()



class CertificateAuthenticationTest(unittest.IsolatedAsyncioTestCase):
    """The credential must actually open a sandbox configured like a real one.

    The server here is set up the way cloud-init sets up a sandbox: it trusts
    the organization's authority and accepts one principal. Nothing else is
    installed, so this fails if the SDK ever falls back to presenting a bare
    key instead of a certificate.
    """

    async def test_one_certificate_opens_any_sandbox_in_the_organization(self) -> None:
        ca = asyncssh.generate_private_key("ssh-ed25519")
        principal = "thunder-org-org-1"
        authorized = asyncssh.import_authorized_keys(
            f'cert-authority,principals="{principal}" '
            + ca.export_public_key().decode().strip()
            + "\n"
        )

        async def handler(process):
            process.stdout.write("ok\n")
            process.exit(0)

        servers = []
        for _ in range(2):  # two sandboxes, one credential
            servers.append(
                await asyncssh.listen(
                    "127.0.0.1", 0,
                    server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
                    authorized_client_keys=authorized,
                    process_factory=handler,
                )
            )
        try:
            with tempfile.TemporaryDirectory() as directory:
                client = AsyncClient(config(directory))

                async def issue(method, path, body=None, query=None):
                    signed = ca.generate_user_certificate(
                        asyncssh.import_public_key(body["ssh_public_key"]),
                        "thunder", principals=[principal],
                        valid_before=int(time.time() + 3600),
                    )
                    return {
                        "ssh_certificate": signed.export_certificate().decode().strip(),
                        "expires_at": datetime.fromtimestamp(
                            time.time() + 3600, timezone.utc
                        ).isoformat(),
                    }

                client._request = mock.AsyncMock(side_effect=issue)  # type: ignore[method-assign]
                credential = await client._credentials.ensure(client)

                for server in servers:
                    port = server.sockets[0].getsockname()[1]
                    async with asyncssh.connect(
                        "127.0.0.1", port=port, username=principal,
                        client_keys=[(credential.key, credential.certificate)],
                        known_hosts=None,
                    ) as connection:
                        result = await connection.run("ignored", check=True)
                        self.assertEqual(result.stdout.strip(), "ok")
                # One certificate, two sandboxes, one call to Thunder.
                self.assertEqual(client._request.await_count, 1)
                await client.close()
        finally:
            for server in servers:
                server.close()


if __name__ == "__main__":
    unittest.main()
