from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import thunder_sandbox.asynchronous as asynchronous
import thunder_sandbox.synchronous as synchronous
import thunder_sandbox as thunder
from thunder_sandbox.asynchronous.client import Client as AsyncClient
from thunder_sandbox.asynchronous.process import Process as AsyncProcess
from thunder_sandbox.asynchronous.sandbox import Sandbox as AsyncSandbox
from thunder_sandbox._common.config import ClientConfig, DEFAULT_API_URL, ThunderPaths
from thunder_sandbox._common.exceptions import (
    AuthenticationError,
    CapacityError,
    ConnectionError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    SandboxTimeoutError,
)
from thunder_sandbox._common.types import GPUType, SandboxStatus
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

    async def test_ssh_uses_accept_new_and_shared_known_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            command = sandbox.ssh.command
            self.assertIn("StrictHostKeyChecking=accept-new", command)
            self.assertIn(
                f"UserKnownHostsFile={client.config.paths.known_hosts}", command
            )
            self.assertNotIn("StrictHostKeyChecking=no", command)
            self.assertNotIn("UserKnownHostsFile=/dev/null", command)
            await client.close()

    async def test_api_host_key_is_pinned_on_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            prepare_key(client.config.paths)
            response = {
                **SANDBOX_RESPONSE,
                "ssh": {
                    **SANDBOX_RESPONSE["ssh"],
                    "host_key": "ssh-ed25519 AAAATEST",
                },
            }
            sandbox = AsyncSandbox._from_response(client, response)
            self.assertEqual(
                client.config.paths.known_hosts.read_text(encoding="utf-8"),
                "[sandbox.example]:2222 ssh-ed25519 AAAATEST\n",
            )
            self.assertEqual(
                sandbox.ssh.known_hosts_path, client.config.paths.known_hosts
            )
            await client.close()

    async def test_exec_uses_asyncssh_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            process = mock.Mock(
                stdin=mock.Mock(), stdout=mock.Mock(), stderr=mock.Mock(), returncode=0
            )
            process.wait = mock.AsyncMock()
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

    async def test_wait_until_ready_uses_async_polling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client = self.sandbox(directory)
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    {**SANDBOX_RESPONSE, "status": "created", "ssh": None},
                    SANDBOX_RESPONSE,
                ]
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep:
                self.assertIs(await sandbox.wait_until_ready(), sandbox)
            sleep.assert_awaited_once()
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
            client._request = mock.AsyncMock(  # type: ignore[method-assign]
                return_value={"id": "sbx-test"}
            )
            with mock.patch(
                "thunder_sandbox.asynchronous.sandbox._validate_create_options"
            ):
                with self.assertRaisesRegex(InvalidRequestError, "already exists"):
                    await AsyncSandbox.create(
                        ssh_public_key="ssh-ed25519 AAAA",
                        ssh_private_key="PRIVATE",
                        client=client,
                    )
            self.assertEqual(
                [call.args[:2] for call in client._request.await_args_list],
                [
                    ("POST", "/sandboxes/start"),
                    ("POST", "/sandboxes/sbx-test/stop"),
                ],
            )
            await client.close()

    async def test_invalid_private_key_fails_before_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock()  # type: ignore[method-assign]
            with self.assertRaisesRegex(InvalidRequestError, "valid private key"):
                await AsyncSandbox.create(
                    ssh_public_key="ssh-ed25519 AAAA",
                    ssh_private_key="not a private key",
                    client=client,
                )
            client._request.assert_not_awaited()
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
            asynchronous.terminate = mock.AsyncMock()  # type: ignore[method-assign]
            try:
                self.assertIs(sandbox.refresh(), sandbox)
                self.assertIsNone(sandbox.poll())
                self.assertEqual(sandbox.wait(timeout=3), 0)
                self.assertIs(sandbox.wait_until_ready(timeout=4), sandbox)
                sandbox.terminate(timeout=5)
            finally:
                client.close()
            asynchronous.refresh.assert_awaited_once_with()  # type: ignore[attr-defined]
            asynchronous.wait.assert_awaited_once_with(timeout=3)  # type: ignore[attr-defined]
            asynchronous.terminate.assert_awaited_once_with(timeout=5)  # type: ignore[attr-defined]

    def test_exec_and_streams_remain_on_persistent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox, client, asynchronous = self.sandbox(directory)

            class Reader:
                async def read(self, n=-1):
                    return "output"

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
            raw.wait = mock.AsyncMock()
            remote = AsyncProcess(raw)
            asynchronous.exec = mock.AsyncMock(return_value=remote)  # type: ignore[method-assign]
            try:
                process = sandbox.exec("echo", "hello")
                self.assertEqual(process.stdin.write("input"), 5)
                self.assertEqual(process.stdout.read(), "output")
                self.assertEqual(process.wait(), 0)
            finally:
                client.close()

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
                try:
                    self.assertIs(await sandbox.refresh_async(), sandbox)
                    self.assertIsNone(await sandbox.poll_async())
                    self.assertEqual(await sandbox.wait_async(timeout=3), 0)
                    self.assertIs(
                        await sandbox.wait_until_ready_async(timeout=4), sandbox
                    )
                finally:
                    await client.close_async()

        asyncio.run(exercise())


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


if __name__ == "__main__":
    unittest.main()
