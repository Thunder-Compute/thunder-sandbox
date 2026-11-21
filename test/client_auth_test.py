from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from thunder_sandbox._common.config import ClientConfig, ThunderPaths
from thunder_sandbox._common.exceptions import AuthenticationError
from thunder_sandbox.asynchronous.client import Client


def cli_config(directory: str, token: str) -> ClientConfig:
    paths = ThunderPaths(Path(directory))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.credentials.write_text(json.dumps({"token": token}))
    with mock.patch.dict(os.environ, {"TNR_API_TOKEN": ""}):
        return ClientConfig(api_url="https://api.example", paths=paths)


def expired() -> AuthenticationError:
    return AuthenticationError("Authentication required", status=401)


class CLITokenRenewalTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_cli_token_is_read_again_and_the_request_resent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "old")
            client = Client(config)
            sent: list[str] = []

            async def send(*args: object) -> bytes:
                sent.append(config.api_token or "")
                if config.api_token == "old":
                    raise expired()
                return b"{}"

            client._send = send  # type: ignore[method-assign]
            config.paths.credentials.write_text(json.dumps({"token": "new"}))
            self.assertEqual(await client._request("POST", "/sandboxes/start", {}), {})
            self.assertEqual(sent, ["old", "new"])
            await client.close()

    async def test_a_401_retries_when_another_request_already_adopted_the_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "old")
            client = Client(config)
            sent: list[str] = []

            async def send(*_args: object) -> bytes:
                sent.append(config.api_token or "")
                if config.api_token == "old":
                    # Presented "old". A sibling then adopted the token the CLI
                    # wrote, before this handler reloads the file.
                    config.paths.credentials.write_text(json.dumps({"token": "new"}))
                    config.api_token = "new"
                    raise expired()
                return b"{}"

            client._send = send  # type: ignore[method-assign]
            self.assertEqual(await client._request("GET", "/sandboxes"), {})
            self.assertEqual(sent, ["old", "new"])
            await client.close()

    async def test_an_unchanged_or_explicit_token_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "same")
            client = Client(config)
            client._send = mock.AsyncMock(side_effect=expired())  # type: ignore[method-assign]
            with self.assertRaises(AuthenticationError):
                await client._request("GET", "/sandboxes")
            self.assertEqual(client._send.await_count, 1)
            await client.close()

            explicit = ClientConfig(api_url="https://api.example", api_token="given",
                                    paths=ThunderPaths(Path(directory)))
            config.paths.credentials.write_text(json.dumps({"token": "other"}))
            self.assertFalse(explicit.reload_cli_token())
            self.assertEqual(explicit.api_token, "given")

    async def test_forbidden_is_not_an_expired_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "old")
            client = Client(config)
            client._send = mock.AsyncMock(  # type: ignore[method-assign]
                side_effect=AuthenticationError("forbidden", status=403)
            )
            config.paths.credentials.write_text(json.dumps({"token": "new"}))
            with self.assertRaises(AuthenticationError):
                await client._request("GET", "/sandboxes")
            self.assertEqual(client._send.await_count, 1)
            await client.close()


if __name__ == "__main__":
    unittest.main()
