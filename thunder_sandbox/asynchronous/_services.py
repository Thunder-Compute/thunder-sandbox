"""Service readiness and sandbox-owned termination tasks."""

from __future__ import annotations

import asyncio
import logging
import shlex
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._common.exceptions import ConnectionError, SandboxFailedError, SandboxTimeoutError
from .._common.lifecycle import finish_cleanup
from .._common.types import validate_port
from .process import Process

if TYPE_CHECKING:
    from .sandbox import Sandbox

STOP_WAIT_SECONDS = 30.0
logger = logging.getLogger(__name__)


class ServiceManager:
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        self._stops: set[asyncio.Task[None]] = set()
        self._closed = False

    async def start(
        self, *args: str, port: int, ready_timeout: float,
        workdir: str | None, env: Mapping[str, str | None] | None,
    ) -> Process[str]:
        validate_port(port)
        if self._closed:
            raise ConnectionError("sandbox services are closed")
        process = await self._sandbox.exec(
            *args, workdir=workdir, env=env, durable=True, retain=True,
        )
        try:
            await asyncio.wait_for(self._wait_listening(process, port), ready_timeout)
        except asyncio.TimeoutError as exc:
            failure: BaseException = SandboxTimeoutError(
                f"service {process.id} did not listen on port {port} within "
                f"{ready_timeout} seconds; recover logs with sandbox.get_process('{process.id}')"
            )
            failure.__cause__ = exc
        except BaseException as exc:
            failure = exc
        else:
            return process

        # A wait deadline must not cancel a durable stop before it can send a signal.
        stop = asyncio.create_task(process.terminate(), name=f"stop service {process.id}")
        self._stops.add(stop)
        stop.add_done_callback(self._stop_finished)
        try:
            await finish_cleanup(asyncio.wait_for(asyncio.shield(stop), STOP_WAIT_SECONDS))
        except BaseException as cleanup_error:
            raise failure from cleanup_error
        raise failure

    async def _wait_listening(self, process: Process[str], port: int) -> None:
        probe = shlex.join([
            "python3", "-c", "import socket; "
            f"socket.create_connection(('127.0.0.1', {port}), 1).close()",
        ])
        while True:
            code = await process.poll()
            if code is not None:
                raise SandboxFailedError(
                    f"service {process.id} exited with status {code}; "
                    f"recover logs with sandbox.get_process('{process.id}')"
                )
            result = await self._sandbox._run_idempotent_command(
                probe, name=f"service {process.id} readiness", check=False,
            )
            if result.exit_status == 0:
                return
            await asyncio.sleep(0.1)

    def _stop_finished(self, task: asyncio.Task[None]) -> None:
        self._stops.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("%s failed", task.get_name(), exc_info=task.exception())

    async def close(self) -> None:
        """End pending stop observations before their SSH transport is closed."""
        self._closed = True
        stops = tuple(self._stops)
        for stop in stops:
            stop.cancel()
        if stops:
            await asyncio.gather(*stops, return_exceptions=True)
