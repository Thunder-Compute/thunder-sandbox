"""Blocking bridge to a persistent asyncio event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class AsyncBridge:
    """Run coroutines on a dedicated loop and synchronously await their results."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="thunder-sandbox-asyncio",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    @staticmethod
    async def _await(awaitable: Awaitable[T]) -> T:
        return await awaitable

    @staticmethod
    def _close(awaitable: Awaitable[object]) -> None:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()

    def run(self, coroutine: Awaitable[T]) -> T:
        if self._closed or self._loop is None:
            self._close(coroutine)
            raise RuntimeError("synchronous client is closed")
        if threading.current_thread() is self._thread:
            self._close(coroutine)
            raise RuntimeError("cannot call the synchronous API from its event-loop thread")
        future = asyncio.run_coroutine_threadsafe(self._await(coroutine), self._loop)
        return future.result()

    async def run_async(self, coroutine: Awaitable[T]) -> T:
        """Await a coroutine on the bridge loop without blocking the caller's loop."""
        if self._closed or self._loop is None:
            self._close(coroutine)
            raise RuntimeError("client is closed")
        future = asyncio.run_coroutine_threadsafe(self._await(coroutine), self._loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join()
