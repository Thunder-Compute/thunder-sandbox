"""Blocking bridge to a persistent asyncio event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import AsyncIterator, Awaitable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, suppress
import sys
from typing import TypeVar

from .._common.lifecycle import finish_cleanup

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

    def _submit(
        self, coroutine: Awaitable[T],
    ) -> tuple[concurrent.futures.Future[T], concurrent.futures.Future[None]]:
        if self._closed or self._loop is None:
            self._close(coroutine)
            raise RuntimeError("SDK connection is closed")
        # Cancelling the result must not report that native finalizers have finished.
        result: concurrent.futures.Future[T] = concurrent.futures.Future()
        finished: concurrent.futures.Future[None] = concurrent.futures.Future()
        loop = self._loop

        def start() -> None:
            task = loop.create_task(self._await(coroutine))

            def complete(task: asyncio.Task[T]) -> None:
                try:
                    value = task.result()
                except BaseException as exc:
                    if not result.done():
                        with suppress(concurrent.futures.InvalidStateError):
                            result.set_exception(exc)
                else:
                    if not result.done():
                        with suppress(concurrent.futures.InvalidStateError):
                            result.set_result(value)
                finally:
                    self._close(coroutine)
                    finished.set_result(None)

            task.add_done_callback(complete)
            result.add_done_callback(
                lambda future: loop.call_soon_threadsafe(task.cancel) if future.cancelled() else None
            )

        loop.call_soon_threadsafe(start)
        return result, finished

    def run(self, coroutine: Awaitable[T]) -> T:
        if threading.current_thread() is self._thread:
            self._close(coroutine)
            raise RuntimeError("cannot call the synchronous API from its event-loop thread")
        future, finished = self._submit(coroutine)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            finished.result()
            raise

    async def run_async(self, coroutine: Awaitable[T]) -> T:
        """Cancel and join native work before its owning scope can close the loop."""
        future, finished = self._submit(coroutine)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()

            async def join() -> None:
                await asyncio.wrap_future(finished)

            await finish_cleanup(join())
            raise

    @contextmanager
    def context(self, scope: AbstractAsyncContextManager[T]) -> Iterator[T]:
        value = self.run(scope.__aenter__())
        try:
            yield value
        except BaseException:
            if not self.run(scope.__aexit__(*sys.exc_info())):
                raise
        else:
            self.run(scope.__aexit__(None, None, None))

    @asynccontextmanager
    async def context_async(self, scope: AbstractAsyncContextManager[T]) -> AsyncIterator[T]:
        value = await self.run_async(scope.__aenter__())
        try:
            yield value
        except BaseException:
            if not await self.run_async(scope.__aexit__(*sys.exc_info())):
                raise
        else:
            await self.run_async(scope.__aexit__(None, None, None))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join()
