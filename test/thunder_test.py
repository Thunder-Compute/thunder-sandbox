from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
import time
import threading
import unittest
from pathlib import Path
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
from thunder_sandbox.asynchronous.sandbox import WAIT_WINDOW_MAX_SECONDS
from thunder_sandbox.asynchronous.sandbox import Sandbox as AsyncSandbox
from thunder_sandbox.asynchronous.sandbox import _pinned_host_key
from thunder_sandbox.synchronous._bridge import AsyncBridge
from thunder_sandbox.synchronous.client import Client
from thunder_sandbox.synchronous.process import Process
from thunder_sandbox.synchronous.sandbox import Sandbox

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
    "expires_at": "2026-08-23T13:00:00Z",
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


class ConfigTest(unittest.TestCase):
    def test_public_distribution_exports_both_apis(self) -> None:
        self.assertIs(thunder.Client, Client)
        self.assertIs(thunder.Sandbox, Sandbox)
        self.assertIs(thunder.Process, Process)
        self.assertIs(synchronous.Client, Client)
        self.assertIs(synchronous.Sandbox, Sandbox)
        self.assertIs(synchronous.Process, Process)
        self.assertIs(asynchronous.Client, AsyncClient)
        self.assertIs(asynchronous.Sandbox, AsyncSandbox)
        self.assertIs(asynchronous.Process, AsyncProcess)
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
                "get_sandbox_by_name", "list_sandboxes",
            ),
            thunder.Sandbox: (
                "create",
                "download",
                "exec",
                "from_id",
                "from_name",
                "poll",
                "refresh",
                "terminate",
                "update_network_policy",
                "upload",
                "wait",
                "wait_until_ready",
            ),
            thunder.Process: ("poll", "terminate", "wait"),
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


class AsyncSandboxTest(unittest.IsolatedAsyncioTestCase):
    def sandbox(self, directory: str) -> tuple[AsyncSandbox, AsyncClient]:
        client = AsyncClient(config(directory))
        prepare_key(client.config.paths)
        return AsyncSandbox._from_response(client, SANDBOX_RESPONSE), client

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
                self.assertEqual(await remote.wait(), 0)
            connection.create_process.assert_awaited_once_with(
                "echo hello", encoding=None, term_type="xterm"
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
                self.assertEqual(process.stdin.write("input"), 5)
                self.assertEqual(process.stdout.read(), "output")
                self.assertEqual(process.wait(), 0)
            finally:
                client.close()

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
            os.chmod(root, 0o500)
            try:
                credential = await client._credentials.ensure(client)
                self.assertTrue(credential.is_usable())
                self.assertFalse(client.config.paths.ssh_key.exists())
            finally:
                os.chmod(root, 0o700)
                await client.close()

    async def test_a_refused_certificate_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock(return_value={})  # type: ignore[method-assign]
            with self.assertRaises(SandboxError):
                await client._credentials.ensure(client)
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
