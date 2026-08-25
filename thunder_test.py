from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from thunder.client import Client, USER_AGENT
from thunder.config import ClientConfig, DEFAULT_API_URL, ThunderPaths
from thunder.exceptions import ConnectionError as ThunderConnectionError
from thunder.exceptions import InvalidRequestError, SandboxFailedError, SandboxTimeoutError
from thunder.sandbox import Sandbox


SANDBOX_RESPONSE = {
    "name": "sbx-test", "status": "running",
    "spec": {"cpu_count": 4, "memory_gib": 32, "storage_gib": 50},
    "network_policy": {"internet_access": "open", "domain_allowlist": ["*.thundercompute.com"]},
    "created_at": "2026-08-23T12:00:00Z",
    "ssh": {"host": "sandbox.example", "port": 2222, "user": "ubuntu"},
}


class FakeClient(Client):
    def __init__(self, paths: ThunderPaths) -> None:
        super().__init__(ClientConfig(api_url="https://api.example", api_token="token", paths=paths))
        self.requests: list[tuple[str, str, object | None]] = []

    def _request(self, method: str, path: str, body: object | None = None, query=None):
        self.requests.append((method, path, body))
        return {"name": "sbx-test"} if path == "/sandboxes/start" else SANDBOX_RESPONSE


class ConfigTest(unittest.TestCase):
    def test_cli_config_and_environment_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            paths.credentials.write_text(json.dumps({"token": "file-token", "api_url": "https://file"}))
            with mock.patch.dict(os.environ, {"TNR_API_TOKEN": "env-token", "TNR_API_URL": "https://env"}, clear=False):
                config = ClientConfig(paths=paths)
            self.assertEqual(config.api_token, "env-token")
            self.assertEqual(config.api_url, "https://env")
            self.assertEqual(ClientConfig(api_token="x", paths=ThunderPaths(Path(directory) / "missing")).api_url, DEFAULT_API_URL)


class RequestHeaderTest(unittest.TestCase):
    def test_requests_send_an_explicit_user_agent(self) -> None:
        # urllib's default "Python-urllib/<version>" is rejected by edge bot
        # rules in front of the API, so the SDK must name itself.
        captured: dict[str, str] = {}

        class Response:
            def read(self) -> bytes:
                return b"{}"

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> bool:
                return False

        def urlopen(request: object, timeout: float | None = None) -> Response:
            captured.update(dict(request.header_items()))  # type: ignore[attr-defined]
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            client = Client(ClientConfig(api_url="https://api.example", api_token="token", paths=ThunderPaths(Path(directory))))
            with mock.patch("thunder.client.urllib.request.urlopen", urlopen):
                client._request("GET", "/sandboxes")

        self.assertEqual(captured["User-agent"], USER_AGENT)
        self.assertNotIn("python-urllib", captured["User-agent"].lower())
        self.assertEqual(captured["Thunder-client"], "PYTHON-SDK")


class SandboxTest(unittest.TestCase):
    def test_generated_key_exists_before_start_and_moves_to_sandbox_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            client = FakeClient(paths)

            def generate(path: Path) -> None:
                path.write_text("PRIVATE")
                path.with_suffix(".pub").write_text("ssh-ed25519 GENERATED")

            with mock.patch("thunder.sandbox._generate_key_pair", side_effect=generate):
                Sandbox.create(client=client)

            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(request["ssh_public_key"], "ssh-ed25519 GENERATED")
            self.assertEqual(paths.sandbox_private_key("sbx-test").read_text(), "PRIVATE")
            self.assertFalse(any(path.name.startswith(".creating-") for path in paths.sandbox_keys.iterdir()))

    def test_explicit_pair_is_sent_and_stored_by_returned_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            client = FakeClient(paths)
            sandbox = Sandbox.create(ssh_public_key="ssh-ed25519 PUBLIC", ssh_private_key="PRIVATE", client=client)
            self.assertEqual(sandbox.id, "sbx-test")
            self.assertEqual(paths.sandbox_private_key("sbx-test").read_text(), "PRIVATE")
            self.assertEqual(paths.sandbox_private_key("sbx-test").stat().st_mode & 0o777, 0o600)
            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(request["ssh_public_key"], "ssh-ed25519 PUBLIC")
            self.assertEqual(request["network_policy"], {
                "internet_access": "restricted",
                "cidr_allowlist": ["0.0.0.0/0"],
                "domain_allowlist": ["*"],
            })

    def test_block_network_rejects_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidRequestError):
                Sandbox.create(
                    block_network=True,
                    outbound_domain_allowlist=["example.com"],
                    client=FakeClient(ThunderPaths(Path(directory))),
                )

    def test_single_restriction_keeps_the_other_gate_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(ThunderPaths(Path(directory)))
            with mock.patch("thunder.sandbox._generate_key_pair") as generate:
                generate.side_effect = lambda path: (
                    path.write_text("PRIVATE"),
                    path.with_suffix(".pub").write_text("ssh-ed25519 GENERATED"),
                )
                Sandbox.create(outbound_domain_allowlist=["example.com"], client=client)
            request = client.requests[0][2]
            assert isinstance(request, dict)
            self.assertEqual(request["network_policy"]["cidr_allowlist"], ["0.0.0.0/0"])
            self.assertEqual(request["network_policy"]["domain_allowlist"], ["example.com"])

    def test_key_pair_must_be_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidRequestError):
                Sandbox.create(ssh_public_key="public", client=FakeClient(ThunderPaths(Path(directory))))

    def test_missing_immutable_key_prevents_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Sandbox._from_response(FakeClient(ThunderPaths(Path(directory))), SANDBOX_RESPONSE)
            with self.assertRaises(SandboxFailedError):
                _ = sandbox.ssh

    def test_exec_uses_sandbox_key_and_argument_quoting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ThunderPaths(Path(directory))
            paths.sandbox_keys.mkdir(parents=True)
            paths.sandbox_private_key("sbx-test").write_text("PRIVATE")
            sandbox = Sandbox._from_response(FakeClient(paths), SANDBOX_RESPONSE)
            process = mock.Mock(stdin=mock.Mock(), stdout=mock.Mock(), stderr=mock.Mock(), pid=42, returncode=None)
            with mock.patch("thunder.sandbox.subprocess.Popen", return_value=process) as popen:
                sandbox.exec("printf", "%s", "hello world", workdir="/tmp/my dir")
            command = popen.call_args.args[0]
            self.assertEqual(command[0], "ssh")
            self.assertIn(str(paths.sandbox_private_key("sbx-test")), command)
            self.assertEqual(command[-1], "cd '/tmp/my dir' && printf %s 'hello world'")


if __name__ == "__main__":
    unittest.main()


class SandboxWaitTest(unittest.TestCase):
    """Waiting spans minutes, so one slow response must not end the wait."""

    def _sandbox(self, directory: str, statuses: list[object]) -> Sandbox:
        """Build a sandbox, then script what each later refresh returns."""
        client = FakeClient(ThunderPaths(Path(directory)))
        sandbox = Sandbox.from_id("sbx-test", client=client)
        responses = list(statuses)

        def fake_request(method, path, body=None, query=None):
            client.requests.append((method, path, body))
            nextval = responses.pop(0) if responses else "running"
            if isinstance(nextval, Exception):
                raise nextval
            return {**SANDBOX_RESPONSE, "status": nextval}

        client._request = fake_request  # type: ignore[method-assign]
        return sandbox

    def test_wait_until_running_retries_transient_connection_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [
                ThunderConnectionError("read timed out"),
                ThunderConnectionError("read timed out"),
                "running",
            ])
            with mock.patch("thunder.sandbox.time.sleep"):
                self.assertIs(sandbox.wait_until_running(timeout=30), sandbox)

    def test_wait_until_running_surfaces_an_outage_without_waiting_out_the_timeout(self) -> None:
        # A real outage must not hide behind a long deadline: the connection
        # error itself is raised once failures run past the grace window.
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [ThunderConnectionError("down")] * 500)
            clock = iter(float(i) for i in range(0, 100000, 5))
            with mock.patch("thunder.sandbox.time.sleep"), \
                 mock.patch("thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)):
                with self.assertRaises(ThunderConnectionError):
                    # Deadline is an hour away; the grace window is what fires.
                    sandbox.wait_until_running(timeout=3600)

    def test_wait_until_running_forgives_blips_separated_by_success(self) -> None:
        # Failures that do not run consecutively must never accumulate into an
        # outage, however long the wait goes on.
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [
                ThunderConnectionError("blip"), "pending",
                ThunderConnectionError("blip"), "pending",
                ThunderConnectionError("blip"), "running",
            ])
            clock = iter(float(i) for i in range(0, 100000, 20))
            with mock.patch("thunder.sandbox.time.sleep"), \
                 mock.patch("thunder.sandbox.time.monotonic", side_effect=lambda: next(clock)):
                self.assertIs(sandbox.wait_until_running(timeout=3600), sandbox)

    def test_wait_until_running_still_reports_a_failed_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [ThunderConnectionError("blip"), "failed"])
            with mock.patch("thunder.sandbox.time.sleep"):
                with self.assertRaises(SandboxFailedError):
                    sandbox.wait_until_running(timeout=30)

    def test_wait_retries_transient_connection_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sandbox = self._sandbox(directory, [ThunderConnectionError("blip"), "stopped"])
            with mock.patch("thunder.sandbox.time.sleep"):
                self.assertEqual(sandbox.wait(timeout=30), 0)
