from __future__ import annotations

import asyncio
import json
import os
import tempfile
import tarfile
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
    RateLimitError,
    SandboxError,
    SandboxFailedError,
    SandboxTimeoutError,
    ServiceUnavailableError,
    UnsupportedFeatureError,
)
from thunder_sandbox._common.types import GPUType, SandboxStatus
from thunder_sandbox.asynchronous.client import USER_AGENT
from thunder_sandbox.asynchronous.client import Client as AsyncClient
from thunder_sandbox.asynchronous.process import Process as AsyncProcess
from thunder_sandbox.asynchronous.sandbox import Sandbox as AsyncSandbox
from thunder_sandbox.asynchronous.sandbox import _pinned_host_key
from thunder_sandbox.image import Image, ResolvedImage, _create_canonical_build_context
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


def canonical_context(root: Path):
    return _create_canonical_build_context(
        root, root / ".thunder" / "image_build_contexts"
    )


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


class ImageTest(unittest.TestCase):
    def test_registry_image_accepts_public_or_complete_private_credentials(
        self,
    ) -> None:
        public = Image.from_registry("ubuntu:24.04")
        private = Image.from_registry(
            "registry.example.com/team/image:latest",
            username="user",
            password="secret",
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
                    "private.example/image:latest", "user", "secret"
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
                    resolved = await client.resolve_image(Image.from_dockerfile(root))
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

    async def test_image_is_wired_but_rejected_until_the_api_supports_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = AsyncClient(config(directory))
            client._request = mock.AsyncMock()  # type: ignore[method-assign]
            with self.assertRaises(UnsupportedFeatureError):
                await AsyncSandbox.create(
                    image=Image.from_registry("ubuntu:24.04"),
                    client=client,
                )
            client._request.assert_not_awaited()
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
