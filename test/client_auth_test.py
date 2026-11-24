from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aiohttp

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

    async def test_concurrent_refusals_all_resend_with_the_renewed_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "old")
            client = Client(config)
            both_sent = asyncio.Event()
            in_flight = 0
            sent: list[str] = []

            async def send(*args: object) -> bytes:
                nonlocal in_flight
                token = config.api_token or ""
                sent.append(token)
                if token == "old":
                    in_flight += 1
                    if in_flight == 2:
                        both_sent.set()
                    await both_sent.wait()  # both carry the old token
                    raise expired()
                return b"{}"

            client._send = send  # type: ignore[method-assign]
            config.paths.credentials.write_text(json.dumps({"token": "new"}))
            results = await asyncio.gather(
                client._request("GET", "/sandboxes"),
                client._request("GET", "/sandboxes"),
            )
            self.assertEqual(results, [{}, {}])
            self.assertEqual(sorted(sent), ["new", "new", "old", "old"])
            await client.close()

    async def test_the_resend_after_renewal_keeps_connection_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = cli_config(directory, "old")
            client = Client(config)
            outcomes: list[object] = [expired(), aiohttp.ClientConnectionError("blip"), b"{}"]
            timeouts: list[float | None] = []

            async def send(*args: object) -> bytes:
                timeouts.append(args[-1])  # type: ignore[arg-type]
                outcome = outcomes.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome  # type: ignore[return-value]

            client._send = send  # type: ignore[method-assign]
            config.paths.credentials.write_text(json.dumps({"token": "new"}))
            clock = iter(range(0, 1000, 5))  # each reading is 5 s after the last
            with mock.patch("asyncio.sleep", mock.AsyncMock()), mock.patch(
                "thunder_sandbox.asynchronous.client.time.monotonic", lambda: next(clock)
            ):
                self.assertEqual(await client._request("GET", "/sandboxes", timeout=30), {})
            self.assertEqual(len(timeouts), 3)
            self.assertEqual(timeouts[0], 30)
            self.assertLess(timeouts[1], 30)  # what is left, not the original
            self.assertLess(timeouts[2], timeouts[1])
            await client.close()

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
