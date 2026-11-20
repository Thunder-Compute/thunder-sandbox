from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from thunder_sandbox._channels import (
    SUPPORTED,
    ExecHandle,
    ExecutionChannel,
    Fixed,
    Unsupported,
    optional_fields,
)
from thunder_sandbox._channels._spec import (
    DownloadRequest,
    ExecRequest,
    ExecState,
    ExecStatus,
    ExitReason,
    Identity,
    OutputMode,
    Retention,
    Signal,
    TerminationPolicy,
    TransferOptions,
    TransferResult,
    UploadRequest,
)
from thunder_sandbox._common.exceptions import (
    InvalidRequestError,
    NotFoundError,
    UnsupportedFeatureError,
)

EXEC_ID = "0" * 32


def table(request_type: type, **rules: object) -> dict[str, object]:
    """Everything supported, except the dotted paths given (``__`` for ``.``)."""
    result: dict[str, object] = {path: SUPPORTED for path in optional_fields(request_type)}
    result.update({key.replace("__", "."): rule for key, rule in rules.items()})
    return result


class FakeChannel(ExecutionChannel):
    name = "fake"
    exec_support = table(
        ExecRequest,
        umask=Fixed(0o22),
        termination__signal=Fixed(Signal.SIGTERM),
        identity__group=Unsupported("only a user can be chosen"),
        env=Unsupported(),
    )
    upload_support = table(UploadRequest, owner__user=Fixed(0), owner__group=Fixed(0))
    download_support = table(DownloadRequest)

    def __init__(self) -> None:
        self.started: dict[str, FakeHandle] = {}

    async def _exec(self, request: ExecRequest) -> ExecHandle:
        handle = self.started.setdefault(request.exec_id, FakeHandle(request.exec_id, b"a\nb"))
        return handle

    async def _attach(self, exec_id: str) -> ExecHandle:
        try:
            return self.started[exec_id]
        except KeyError:
            raise NotFoundError(exec_id) from None

    async def _upload(self, request: UploadRequest) -> TransferResult:
        return TransferResult(request.destination)

    async def _download(self, request: DownloadRequest) -> TransferResult:
        return TransferResult(str(request.local_path))

    async def after_network_policy_change(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeHandle(ExecHandle):
    def __init__(self, exec_id: str, output: bytes) -> None:
        self._exec_id = exec_id
        self._output = output
        self.stdin = b""

    @property
    def exec_id(self) -> str:
        return self._exec_id

    async def stdout(self) -> AsyncIterator[bytes]:
        for line in self._output.splitlines(keepends=True):
            yield line

    async def stderr(self) -> AsyncIterator[bytes]:
        return
        yield b""

    async def write_stdin(self, data: bytes) -> None:
        self.stdin += data

    async def close_stdin(self) -> None:
        return None

    async def poll(self) -> ExecStatus:
        return ExecStatus(self._exec_id, ExecState.RUNNING)

    async def wait(self) -> ExecStatus:
        return ExecStatus(
            self._exec_id, ExecState.ENDED, reason=ExitReason.EXITED, returncode=0
        )

    async def terminate(self) -> ExecStatus:
        return await self.wait()

    async def cleanup(self) -> None:
        return None


class SupportTableTest(unittest.TestCase):
    def test_exec_fields_are_named_by_leaf(self) -> None:
        fields = optional_fields(ExecRequest)
        self.assertIn("identity.user", fields)
        self.assertIn("termination.graceful_termination_timeout_ms", fields)
        self.assertIn("pty", fields)
        for core in ("exec_id", "executable", "args"):
            self.assertNotIn(core, fields)

    def test_transfer_fields(self) -> None:
        self.assertEqual(
            sorted(optional_fields(UploadRequest)),
            sorted(
                [
                    "options.recursive",
                    "options.contents_only",
                    "options.on_conflict",
                    "options.create_parents",
                    "options.preserve_times",
                    "options.symlinks",
                    "options.timeout_ms",
                    "owner.user",
                    "owner.group",
                ]
            ),
        )

    def test_incomplete_table_is_refused_at_definition(self) -> None:
        with self.assertRaisesRegex(TypeError, "missing umask"):
            class Broken(FakeChannel):
                exec_support = {k: v for k, v in FakeChannel.exec_support.items() if k != "umask"}

    def test_unknown_field_is_refused_at_definition(self) -> None:
        with self.assertRaisesRegex(TypeError, "unknown no_such_field"):
            class Broken(FakeChannel):
                exec_support = {**FakeChannel.exec_support, "no_such_field": SUPPORTED}


class ChannelCheckTest(unittest.IsolatedAsyncioTestCase):
    async def test_unset_fields_pass(self) -> None:
        channel = FakeChannel()
        handle = await channel.exec(ExecRequest(EXEC_ID, "true"))
        self.assertEqual(handle.exec_id, EXEC_ID)
        self.assertEqual(len(channel.started), 1)

    async def test_handle_streams_wait_and_attach(self) -> None:
        channel = FakeChannel()
        handle = await channel.exec(ExecRequest(EXEC_ID, "cat"))
        self.assertEqual(b"".join([chunk async for chunk in handle.stdout()]), b"a\nb")
        self.assertEqual([chunk async for chunk in handle.stderr()], [])
        await handle.write_stdin(b"x")
        self.assertEqual((await handle.wait()).returncode, 0)
        self.assertIs(await channel.attach(EXEC_ID), handle)
        with self.assertRaises(NotFoundError):
            await channel.attach("1" * 32)

    async def test_fixed_value_passes_and_others_fail(self) -> None:
        channel = FakeChannel()
        await channel.exec(ExecRequest(EXEC_ID, "true", umask=0o22))
        with self.assertRaisesRegex(UnsupportedFeatureError, "umask"):
            await channel.exec(ExecRequest(EXEC_ID, "true", umask=0o77))
        with self.assertRaisesRegex(UnsupportedFeatureError, "termination.signal"):
            await channel.exec(
                ExecRequest(EXEC_ID, "true", termination=TerminationPolicy(signal=Signal.SIGKILL))
            )

    async def test_unsupported_nested_and_collection_fields(self) -> None:
        channel = FakeChannel()
        with self.assertRaisesRegex(UnsupportedFeatureError, "only a user can be chosen"):
            await channel.exec(ExecRequest(EXEC_ID, "id", identity=Identity(user=1000, group=1000)))
        await channel.exec(ExecRequest(EXEC_ID, "id", identity=Identity(user=1000)))
        await channel.exec(ExecRequest(EXEC_ID, "true", env={}))
        with self.assertRaisesRegex(UnsupportedFeatureError, "env"):
            await channel.exec(ExecRequest(EXEC_ID, "true", env={"A": "1"}))

    async def test_upload_owner_fixed(self) -> None:
        channel = FakeChannel()
        await channel.upload(UploadRequest(Path("x"), "/tmp/x"))
        await channel.upload(UploadRequest(Path("x"), "/tmp/x", owner=Identity(user=0, group=0)))
        with self.assertRaisesRegex(UnsupportedFeatureError, "owner.user"):
            await channel.upload(UploadRequest(Path("x"), "/tmp/x", owner=Identity(user=1000)))


class SpecValidationTest(unittest.TestCase):
    def test_exec_request_rules(self) -> None:
        cases = [
            dict(exec_id="nope", executable="true"),
            dict(exec_id=EXEC_ID, executable=""),
            dict(exec_id=EXEC_ID, executable="true", args=("a\x00b",)),
            dict(exec_id=EXEC_ID, executable="true", working_dir="relative"),
            dict(exec_id=EXEC_ID, executable="true", env={"A=B": "1"}),
            dict(exec_id=EXEC_ID, executable="true", env={"A": "1"}, unset_env=("A",)),
            dict(exec_id=EXEC_ID, executable="sh", pty=True, stderr=OutputMode.DISCARD),
            dict(exec_id=EXEC_ID, executable="sh", pty=True, retention=Retention.RETAIN),
            dict(exec_id=EXEC_ID, executable="true", umask=0o1000),
            dict(exec_id=EXEC_ID, executable="true", inherit_env="yes"),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidRequestError):
                ExecRequest(**kwargs)

    def test_exec_request_is_immutable(self) -> None:
        env = {"A": "1"}
        request = ExecRequest(EXEC_ID, "true", env=env)
        env["B"] = "2"
        self.assertEqual(dict(request.env), {"A": "1"})
        with self.assertRaises(TypeError):
            request.env["C"] = "3"  # type: ignore[index]

    def test_identity_rules(self) -> None:
        Identity(user="root", group=0)
        for kwargs in (dict(user=True), dict(user=-1), dict(user="a:b"), dict(group=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidRequestError):
                Identity(**kwargs)

    def test_status_consistency(self) -> None:
        ExecStatus(EXEC_ID, ExecState.ENDED, reason=ExitReason.TERMINATED, returncode=143)
        with self.assertRaises(ValueError):
            ExecStatus(EXEC_ID, ExecState.RUNNING, returncode=0)
        with self.assertRaises(ValueError):
            ExecStatus(EXEC_ID, ExecState.ENDED, reason=ExitReason.EXITED)

    def test_interrupted_has_no_returncode(self) -> None:
        status = ExecStatus(EXEC_ID, ExecState.ENDED, reason=ExitReason.INTERRUPTED)
        self.assertIsNone(status.returncode)
        with self.assertRaises(ValueError):
            ExecStatus(EXEC_ID, ExecState.ENDED, reason=ExitReason.INTERRUPTED, returncode=137)

    def test_transfer_options_rules(self) -> None:
        for kwargs in (dict(timeout_ms=0), dict(contents_only=True), dict(create_parents="no")):
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidRequestError):
                TransferOptions(**kwargs)
        TransferOptions(recursive=True, contents_only=True)

    def test_upload_destination_must_be_absolute(self) -> None:
        with self.assertRaises(InvalidRequestError):
            UploadRequest(Path("x"), "relative")


if __name__ == "__main__":
    unittest.main()
