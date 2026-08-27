from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from datetime import timezone
from pathlib import Path
from unittest import mock

from thunder.client import Client, USER_AGENT
from thunder.config import ClientConfig, DEFAULT_API_URL, ThunderPaths
from thunder.exceptions import (
    AuthenticationError,
    ConflictError,
    ConnectionError as ThunderConnectionError,
    InvalidRequestError,
    NotFoundError,
    SandboxFailedError,
    SandboxTimeoutError,
    ThunderError,
    UnsupportedFeatureError,
)
from thunder.process import ContainerProcess
from thunder.sandbox import AsyncSandbox, Sandbox
from thunder.types import GPUType, SandboxStatus
import thunder_sandbox


SANDBOX_RESPONSE = {
    "id": "sbx-test",
    "name": "sbx-test",
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


class FakeClient(Client):
    def __init__(self, paths: ThunderPaths) -> None:
        super().__init__(
            ClientConfig(
                api_url="https://api.example",
                api_token="token",
                paths=paths,
            )
        )
        self.requests: list[
            tuple[str, str, object | None, dict[str, object] | None]
        ] = []

    def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append((method, path, body, query))
        if path == "/sandboxes/start":
            return {"id": "sbx-test", "name": "sbx-test"}
        return dict(SANDBOX_RESPONSE)


class HTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "HTTPResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def write_generated_key(path: Path) -> None:
    path.write_text("PRIVATE", encoding="utf-8")
    path.with_suffix(".pub").write_text("ssh-ed25519 GENERATED", encoding="utf-8")


class ConfigTest(unittest.TestCase):
    def test_distribution_import_name_exposes_public_api(self) -> None:
        self.assertIs(thunder_sandbox.Sandbox, Sandbox)
        self.assertIs(thunder_sandbox.Client, Client)
        self.assertIn("Sandbox", thunder_sandbox.__all__)

    def test_configuration_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ThunderPaths(root / "state")
            paths.root.mkdir()
            paths.credentials.write_text(
                json.dumps({"token": "file-token", "api_url": "https://file"}),
                encoding="utf-8",
            )
            (root / ".thunder.json").write_text(
                json.dumps({"api_url": "https://project"}), encoding="utf-8"
            )

            with mock.patch("thunder.config.Path.cwd", return_value=root), mock.patch.dict(
                os.environ,
                {"TNR_API_TOKEN": "env-token", "TNR_API_URL": "https://env"},
                clear=True,
            ):
                environment = ClientConfig(paths=paths)
                explicit = ClientConfig(
                    api_url="https://explicit/",
                    api_token="explicit-token",
                    paths=paths,
                )

            self.assertEqual(environment.api_token, "env-token")
            self.assertEqual(environment.api_url, "https://env")
            self.assertEqual(explicit.api_token, "explicit-token")
            self.assertEqual(explicit.api_url, "https://explicit")

    def test_project_url_precedes_cli_url_and_default_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ThunderPaths(root / "state")
            paths.root.mkdir()
            paths.credentials.write_text(
                json.dumps({"token": "token", "api_url": "https://file"}),
                encoding="utf-8",
            )
            (root / ".thunder.json").write_text(
                json.dumps({"api_url": "https://project"}), encoding="utf-8"
            )
            with mock.patch("thunder.config.Path.cwd", return_value=root), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(ClientConfig(paths=paths).api_url, "https://project")
            with mock.patch(
                "thunder.config.Path.cwd", return_value=root / "empty-project"
            ), mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    ClientConfig(
                        api_token="token", paths=ThunderPaths(root / "missing")
                    ).api_url,
                    DEFAULT_API_URL,
                )

    def test_invalid_configuration_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            paths.credentials.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(InvalidRequestError, "could not read"):
                ClientConfig(paths=paths)

    def test_missing_token_is_an_authentication_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            config = ClientConfig(paths=ThunderPaths(Path(directory)))
            with self.assertRaises(AuthenticationError):
                Client(config)

    def test_paths_reject_unsafe_sandbox_names(self) -> None:
        paths = ThunderPaths(Path("/tmp/thunder-test"))
        for name in ("", ".", "..", "a/b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(InvalidRequestError):
                paths.sandbox_private_key(name)


class ClientRequestTest(unittest.TestCase):
    def _client(self, directory: str) -> Client:
        return Client(
            ClientConfig(
                api_url="https://api.example/root/",
                api_token="secret",
                paths=ThunderPaths(Path(directory)),
            )
        )

    def test_request_serializes_query_body_and_headers(self) -> None:
        captured: dict[str, object] = {}

        def urlopen(request: object, timeout: float | None = None) -> HTTPResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["method"] = request.get_method()  # type: ignore[attr-defined]
            captured["body"] = request.data  # type: ignore[attr-defined]
            captured["headers"] = {
                key.lower(): value
                for key, value in request.header_items()  # type: ignore[attr-defined]
            }
            captured["timeout"] = timeout
            return HTTPResponse(b'{"ok": true}')

        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            with mock.patch("thunder.client.urllib.request.urlopen", urlopen):
                response = client._request(
                    "POST",
                    "/sandboxes/start",
                    {"cpu": 4},
                    {"limit": 100, "page_token": "next token", "empty": ""},
                )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(
            captured["url"],
            "https://api.example/root/v1/sandboxes/start?limit=100&page_token=next+token",
        )
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(json.loads(captured["body"]), {"cpu": 4})  # type: ignore[arg-type]
        headers = captured["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["authorization"], "Bearer secret")
        self.assertEqual(headers["user-agent"], USER_AGENT)
        self.assertEqual(headers["thunder-client"], "PYTHON-SDK")
        self.assertEqual(captured["timeout"], 30)

    def test_http_statuses_map_to_public_exceptions(self) -> None:
        cases = {
            400: InvalidRequestError,
            401: AuthenticationError,
            403: AuthenticationError,
            404: NotFoundError,
            409: ConflictError,
            500: ThunderError,
        }
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            for status, error_type in cases.items():
                error = urllib.error.HTTPError(
                    "https://api.example",
                    status,
                    "error",
                    hdrs=None,
                    fp=io.BytesIO(b'{"message":"specific failure"}'),
                )
                with self.subTest(status=status), mock.patch(
                    "thunder.client.urllib.request.urlopen", side_effect=error
                ), self.assertRaisesRegex(error_type, "specific failure"):
                    client._request("GET", "/sandboxes")

    def test_transport_and_malformed_responses_are_connection_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            responses: list[object] = [
                urllib.error.URLError("offline"),
                HTTPResponse(b"not-json"),
                HTTPResponse(b"[]"),
            ]
            for response in responses:
                with self.subTest(response=response), mock.patch(
                    "thunder.client.urllib.request.urlopen",
                    side_effect=response if isinstance(response, Exception) else None,
                    return_value=response if not isinstance(response, Exception) else None,
                ), self.assertRaises(ThunderConnectionError):
                    client._request("GET", "/sandboxes")

    def test_empty_response_and_closed_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            with mock.patch(
                "thunder.client.urllib.request.urlopen",
                return_value=HTTPResponse(b""),
            ):
                self.assertEqual(client._request("DELETE", "/sandboxes/x"), {})
            client.close()
            with self.assertRaisesRegex(ThunderConnectionError, "closed"):
                client._request("GET", "/sandboxes")

    def test_list_sandboxes_consumes_every_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(directory)
            first = {**SANDBOX_RESPONSE, "id": "sbx-one", "name": "first"}
            second = {**SANDBOX_RESPONSE, "id": "sbx-two", "name": "second"}
            request = mock.Mock(
                side_effect=[
                    {"sandboxes": [first], "next_page_token": "page-2"},
                    {"sandboxes": [second], "next_page_token": ""},
                ]
            )
            client._request = request  # type: ignore[method-assign]

            self.assertEqual(
                [sandbox.id for sandbox in client.list_sandboxes()],
                ["sbx-one", "sbx-two"],
            )
            self.assertEqual(
                request.call_args_list,
                [
                    mock.call(
                        "GET",
                        "/sandboxes",
                        query={"limit": 100, "page_token": ""},
                    ),
                    mock.call(
                        "GET",
                        "/sandboxes",
                        query={"limit": 100, "page_token": "page-2"},
                    ),
                ],
            )


class SandboxCreationTest(unittest.TestCase):
    def test_gpu_type_and_count_must_follow_the_public_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            invalid_options = (
                {"gpu_type": GPUType.H100},
                {"gpu_count": 1},
                {"gpu_type": GPUType.A100, "gpu_count": 0},
                {"gpu_type": GPUType.A6000, "gpu_count": 3},
                {"gpu_type": "h100", "gpu_count": 1},
            )
            for options in invalid_options:
                with self.subTest(options=options), self.assertRaises(InvalidRequestError):
                    Sandbox.create(client=client, **options)  # type: ignore[arg-type]

    def test_generated_key_exists_before_start_and_moves_to_sandbox_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            client = FakeClient(paths)
            with mock.patch(
                "thunder.sandbox._generate_key_pair", side_effect=write_generated_key
            ):
                Sandbox.create(client=client)

            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(request["ssh_public_key"], "ssh-ed25519 GENERATED")
            self.assertEqual(
                paths.sandbox_private_key("sbx-test").read_text(encoding="utf-8"),
                "PRIVATE",
            )
            self.assertFalse(
                any(
                    path.name.startswith(".creating-")
                    for path in paths.sandbox_keys.iterdir()
                )
            )

    def test_request_contains_resources_environment_lifetime_and_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            Sandbox.create(
                cpu=8,
                memory=64,
                storage=200,
                gpu_type=GPUType.H100,
                gpu_count=2,
                env={"KEEP": "value", "DROP": None},
                timeout=None,
                outbound_cidr_allowlist=["203.0.113.0/24"],
                outbound_domain_allowlist=["example.com"],
                ssh_public_key="ssh-ed25519 PUBLIC",
                ssh_private_key="PRIVATE",
                client=client,
            )

            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(
                request["spec"],
                {
                    "cpu_count": 8,
                    "memory_gib": 64,
                    "storage_gib": 200,
                    "gpu_type": "H100",
                    "gpu_count": 2,
                },
            )
            self.assertEqual(request["env"], {"KEEP": "value"})
            self.assertEqual(request["lifetime"], {"enforce_ttl": False})
            self.assertEqual(
                request["network_policy"],
                {
                    "internet_access": "restricted",
                    "cidr_allowlist": ["203.0.113.0/24"],
                    "domain_allowlist": ["example.com"],
                },
            )

    def test_default_network_policy_explicitly_opens_both_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            Sandbox.create(
                ssh_public_key="ssh-ed25519 PUBLIC",
                ssh_private_key="PRIVATE",
                client=client,
            )
            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(
                request["network_policy"],
                {
                    "internet_access": "restricted",
                    "cidr_allowlist": ["0.0.0.0/0"],
                    "domain_allowlist": ["*"],
                },
            )

    def test_single_restriction_keeps_the_other_gate_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            Sandbox.create(
                outbound_domain_allowlist=["example.com"],
                ssh_public_key="ssh-ed25519 PUBLIC",
                ssh_private_key="PRIVATE",
                client=client,
            )
            request = client.requests[0][2]
            assert isinstance(request, dict)
            policy = request["network_policy"]
            assert isinstance(policy, dict)
            self.assertEqual(policy["cidr_allowlist"], ["0.0.0.0/0"])
            self.assertEqual(policy["domain_allowlist"], ["example.com"])

    def test_closed_network_has_empty_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            Sandbox.create(
                block_network=True,
                ssh_public_key="ssh-ed25519 PUBLIC",
                ssh_private_key="PRIVATE",
                client=client,
            )
            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(
                request["network_policy"],
                {
                    "internet_access": "closed",
                    "cidr_allowlist": [],
                    "domain_allowlist": [],
                },
            )

    def test_invalid_creation_options_fail_before_request(self) -> None:
        cases = [
            {"name": "customer-name"},
            {"timeout": -1},
            {"ssh_public_key": "public"},
            {
                "block_network": True,
                "outbound_domain_allowlist": ["example.com"],
            },
        ]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                paths = ThunderPaths(Path(directory))
                client = FakeClient(paths)
                with self.assertRaises(
                    (InvalidRequestError, UnsupportedFeatureError)
                ):
                    Sandbox.create(client=client, **options)  # type: ignore[arg-type]
                self.assertEqual(client.requests, [])
                self.assertFalse(paths.sandbox_keys.exists())

    def test_missing_id_in_start_response_is_a_lifecycle_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            client._request = mock.Mock(return_value={})  # type: ignore[method-assign]
            with mock.patch(
                "thunder.sandbox._generate_key_pair", side_effect=write_generated_key
            ), self.assertRaisesRegex(SandboxFailedError, "sandbox ID"):
                Sandbox.create(client=client)

    def test_existing_key_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            paths.sandbox_keys.mkdir(parents=True)
            private_key = paths.sandbox_private_key("sbx-test")
            private_key.write_text("EXISTING", encoding="utf-8")
            with self.assertRaisesRegex(InvalidRequestError, "already exists"):
                Sandbox.create(
                    ssh_public_key="ssh-ed25519 PUBLIC",
                    ssh_private_key="NEW",
                    client=FakeClient(paths),
                )
            self.assertEqual(private_key.read_text(encoding="utf-8"), "EXISTING")


class SandboxOperationTest(unittest.TestCase):
    def _sandbox(self, directory: str, response: dict[str, object] | None = None) -> Sandbox:
        paths = ThunderPaths(Path(directory))
        paths.sandbox_keys.mkdir(parents=True)
        paths.sandbox_private_key("sbx-test").write_text("PRIVATE", encoding="utf-8")
        return Sandbox._from_response(FakeClient(paths), response or SANDBOX_RESPONSE)

    def test_response_is_parsed_into_typed_info(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = {
                **SANDBOX_RESPONSE,
                "spec": {
                    "cpu_count": 8,
                    "memory_gib": 80,
                    "storage_gib": 500,
                    "gpu_type": "H100",
                    "gpu_count": 2,
                },
            }
            sandbox = self._sandbox(directory, response)
            self.assertEqual(sandbox.status, SandboxStatus.READY)
            self.assertEqual(sandbox.id, "sbx-test")
            self.assertEqual(sandbox.name, "sbx-test")
            self.assertEqual(sandbox.info.resources.cpu, 8)
            self.assertEqual(sandbox.info.resources.gpu_type, GPUType.H100)
            self.assertEqual(sandbox.info.resources.gpu_count, 2)
            self.assertEqual(sandbox.info.created_at.tzinfo, timezone.utc)
            self.assertEqual(sandbox.ssh.host, "sandbox.example")
            self.assertEqual(sandbox.ssh.port, 2222)

    def test_failed_response_preserves_public_failure_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, {
                **SANDBOX_RESPONSE,
                "status": "failed",
                "failure_code": "boot_failed",
                "failure": "Guest initialization failed",
            })

            self.assertEqual(sandbox.info.failure_code, "boot_failed")
            self.assertEqual(sandbox.info.failure, "Guest initialization failed")

    def test_missing_immutable_key_prevents_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Sandbox._from_response(
                FakeClient(ThunderPaths(Path(directory))), SANDBOX_RESPONSE
            )
            with self.assertRaises(SandboxFailedError):
                _ = sandbox.ssh

    def test_get_quotes_sandbox_id_as_one_path_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            Sandbox.from_id("sbx/with space", client=client)
            self.assertEqual(client.requests[0][1], "/sandboxes/sbx%2Fwith%20space")
            with self.assertRaises(InvalidRequestError):
                Sandbox.from_id("", client=client)

    def test_exec_uses_key_pty_workdir_environment_and_argument_quoting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory)
            process = mock.Mock(
                stdin=mock.Mock(),
                stdout=mock.Mock(),
                stderr=mock.Mock(),
                pid=42,
                returncode=None,
            )
            with mock.patch(
                "thunder.sandbox.subprocess.Popen", return_value=process
            ) as popen:
                sandbox.exec(
                    "printf",
                    "%s",
                    "hello world",
                    workdir="/tmp/my dir",
                    env={"A": "space value", "REMOVE": None},
                    pty=True,
                )
            command = popen.call_args.args[0]
            self.assertEqual(command[:2], ["ssh", "-tt"])
            self.assertIn(str(sandbox.ssh.private_key_path), command)
            self.assertEqual(
                command[-1],
                "cd '/tmp/my dir' && env A='space value' -u REMOVE printf %s 'hello world'",
            )
            with self.assertRaises(InvalidRequestError):
                sandbox.exec()

    def test_upload_and_download_build_scp_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory)
            with mock.patch("thunder.sandbox.subprocess.run") as run:
                sandbox.upload("local dir", "/remote dir", recursive=True)
                upload = run.call_args.args[0]
                self.assertEqual(upload[0], "scp")
                self.assertIn("-r", upload)
                self.assertEqual(upload[-2:], ["local dir", "ubuntu@sandbox.example:'/remote dir'"])

                sandbox.download("/remote file", "local-file")
                download = run.call_args.args[0]
                self.assertEqual(
                    download[-2:],
                    ["ubuntu@sandbox.example:'/remote file'", "local-file"],
                )

    def test_transfer_failures_have_public_error_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory)
            with mock.patch(
                "thunder.sandbox.subprocess.run", side_effect=FileNotFoundError
            ), self.assertRaises(UnsupportedFeatureError):
                sandbox.upload("local", "/remote")
            with mock.patch(
                "thunder.sandbox.subprocess.run",
                side_effect=subprocess.CalledProcessError(23, ["scp"]),
            ), self.assertRaisesRegex(SandboxFailedError, "status 23"):
                sandbox.download("/remote", "local")

    def test_terminate_requests_stop_and_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory)
            client = sandbox._client
            sandbox.terminate()
            self.assertEqual(
                client.requests[-2][:3],
                ("POST", "/sandboxes/sbx-test/stop", None),
            )
            self.assertEqual(client.requests[-1][1], "/sandboxes/sbx-test")

    def test_ssh_command_uses_only_the_sandbox_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory)
            command = sandbox.ssh_command
            self.assertEqual(command[0], "ssh")
            self.assertIn("IdentitiesOnly=yes", command)
            self.assertIn("IdentityAgent=none", command)
            self.assertEqual(command[-1], "ubuntu@sandbox.example")


class SandboxWaitTest(unittest.TestCase):
    def _sandbox(self, directory: str, statuses: list[object]) -> Sandbox:
        client = FakeClient(ThunderPaths(Path(directory)))
        sandbox = Sandbox.from_id("sbx-test", client=client)
        responses = list(statuses)

        def fake_request(
            method: str,
            path: str,
            body: object | None = None,
            query: dict[str, object] | None = None,
        ) -> dict[str, object]:
            client.requests.append((method, path, body, query))
            next_value = responses.pop(0) if responses else "ready"
            if isinstance(next_value, Exception):
                raise next_value
            return {**SANDBOX_RESPONSE, "status": next_value}

        client._request = fake_request  # type: ignore[method-assign]
        return sandbox

    def test_wait_until_ready_retries_transient_connection_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(
                directory,
                [
                    ThunderConnectionError("read timed out"),
                    ThunderConnectionError("read timed out"),
                    "ready",
                ],
            )
            with mock.patch("thunder.sandbox.time.sleep"):
                self.assertIs(sandbox.wait_until_ready(timeout=30), sandbox)

    def test_terminate_waits_for_ready_before_sending_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, ["created", "ready"])
            sandbox._info = replace(sandbox.info, status=SandboxStatus.CREATED)
            with mock.patch("thunder.sandbox.time.sleep"):
                sandbox.terminate(timeout=30)
            stop_requests = [request for request in sandbox._client.requests if request[:2] == ("POST", "/sandboxes/sbx-test/stop")]
            self.assertEqual(len(stop_requests), 1)

    def test_terminate_skips_stop_for_terminal_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [])
            sandbox._info = replace(sandbox.info, status=SandboxStatus.FAILED)
            request_count = len(sandbox._client.requests)
            sandbox.terminate(timeout=30)
            self.assertEqual(len(sandbox._client.requests), request_count)

    def test_terminate_times_out_without_sending_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, ["created"] * 10)
            sandbox._info = replace(sandbox.info, status=SandboxStatus.CREATED)
            clock = iter([0.0, 1.0, 6.0])
            with mock.patch("thunder.sandbox.time.sleep"), mock.patch(
                "thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)
            ), self.assertRaises(SandboxTimeoutError):
                sandbox.terminate(timeout=5)
            self.assertFalse(any(request[:2] == ("POST", "/sandboxes/sbx-test/stop") for request in sandbox._client.requests))

    def test_wait_until_ready_surfaces_a_sustained_outage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(
                directory, [ThunderConnectionError("down")] * 100
            )
            clock = iter(float(value) for value in range(0, 1000, 5))
            with mock.patch("thunder.sandbox.time.sleep"), mock.patch(
                "thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)
            ), self.assertRaises(ThunderConnectionError):
                sandbox.wait_until_ready(timeout=3600)

    def test_successful_refresh_resets_the_outage_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(
                directory,
                [
                    ThunderConnectionError("blip"),
                    "created",
                    ThunderConnectionError("blip"),
                    "created",
                    ThunderConnectionError("blip"),
                    "ready",
                ],
            )
            clock = iter(float(value) for value in range(0, 1000, 20))
            with mock.patch("thunder.sandbox.time.sleep"), mock.patch(
                "thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)
            ):
                self.assertIs(sandbox.wait_until_ready(timeout=3600), sandbox)

    def test_terminal_states_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed = self._sandbox(directory, ["failed"])
            with self.assertRaises(SandboxFailedError):
                failed.wait_until_ready(timeout=30)

        with tempfile.TemporaryDirectory() as directory:
            finished = self._sandbox(directory, ["finished"])
            self.assertEqual(finished.wait(timeout=30), 0)

        with tempfile.TemporaryDirectory() as directory:
            failed = self._sandbox(directory, ["failed"])
            self.assertEqual(failed.wait(timeout=30), 1)

    def test_wait_timeout_is_stable_and_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, ["created"] * 10)
            clock = iter([0.0, 1.0, 6.0])
            with mock.patch("thunder.sandbox.time.sleep"), mock.patch(
                "thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)
            ), self.assertRaises(SandboxTimeoutError):
                sandbox.wait_until_ready(timeout=5)

    def test_command_timeout_uses_sandbox_timeout_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [])
            process = mock.Mock()
            process.wait.side_effect = subprocess.TimeoutExpired("ssh", 10)
            sandbox._main_process = process
            with self.assertRaises(SandboxTimeoutError):
                sandbox.wait(timeout=10)


class ProcessTest(unittest.TestCase):
    def test_process_delegates_lifecycle_and_default_timeout(self) -> None:
        raw = mock.Mock(
            stdin=mock.Mock(),
            stdout=mock.Mock(),
            stderr=mock.Mock(),
            pid=123,
            returncode=None,
        )
        process = ContainerProcess(raw, timeout=7)
        self.assertEqual(process.id, "123")
        process.wait()
        raw.wait.assert_called_once_with(timeout=7)
        process.terminate()
        raw.terminate.assert_called_once_with()


class AsyncSandboxTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_wrapper_delegates_without_blocking_the_event_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sync_sandbox = SandboxOperationTest()._sandbox(directory)
            async_sandbox = AsyncSandbox(sync_sandbox)
            process = mock.Mock(
                id="42",
                stdin=mock.Mock(),
                stdout=mock.Mock(),
                stderr=mock.Mock(),
            )
            process.wait.return_value = 0
            with mock.patch.object(sync_sandbox, "exec", return_value=process) as execute:
                async_process = await async_sandbox.exec("echo", "hello")
                self.assertEqual(await async_process.wait(), 0)
            execute.assert_called_once_with("echo", "hello")

    async def test_async_refresh_returns_the_same_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sync_sandbox = SandboxOperationTest()._sandbox(directory)
            async_sandbox = AsyncSandbox(sync_sandbox)
            with mock.patch.object(sync_sandbox, "refresh", return_value=sync_sandbox):
                self.assertIs(await async_sandbox.refresh(), async_sandbox)


if __name__ == "__main__":
    unittest.main()
