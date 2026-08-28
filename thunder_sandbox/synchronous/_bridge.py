"""Blocking bridge to a persistent asyncio event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
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
        loop = self._loop
        result: concurrent.futures.Future[T] = concurrent.futures.Future()
        task_holder: list[asyncio.Task[T] | None] = [None]

        def start() -> None:
            if result.cancelled():
                self._close(coroutine)
                return
            task = loop.create_task(self._await(coroutine))
            task_holder[0] = task
            if result.cancelled():
                task.cancel()
                return

            def done(finished: asyncio.Task[T]) -> None:
                if result.cancelled():
                    return
                try:
                    if finished.cancelled():
                        result.cancel()
                    else:
                        exc = finished.exception()
                        if exc is not None:
                            result.set_exception(exc)
                        else:
                            result.set_result(finished.result())
                except concurrent.futures.InvalidStateError:
                    return

            task.add_done_callback(done)

        loop.call_soon_threadsafe(start)
        try:
            return await asyncio.wrap_future(result)
        except asyncio.CancelledError:
            def cancel() -> None:
                task = task_holder[0]
                if task is not None:
                    task.cancel()
                result.cancel()

            loop.call_soon_threadsafe(cancel)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join()
