"""Blocking adapter over the native asynchronous snapshot implementation."""

from __future__ import annotations

from .._common.types import SnapshotInfo, SnapshotStatus
from ..asynchronous.snapshot import Snapshot as NativeSnapshot
from .client import Client


class Snapshot:
    """A durable disk snapshot of a sandbox.

    It holds the source sandbox's whole disk, including the container's
    changes to its image, but not the image itself, which it pins as
    :attr:`image_id`. Restoring it creates a new sandbox with those files, not
    the source's memory, processes, or GPU state. It outlives its source
    sandbox until it is deleted. A snapshot is not an image and cannot be
    passed as ``image=``.
    """

    def __init__(
        self, client: Client, snapshot: NativeSnapshot, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._snapshot = snapshot
        self._owns_client = owns_client

    @staticmethod
    def _from_async(client: Client, snapshot: NativeSnapshot) -> "Snapshot":
        return client._wrap_snapshot(snapshot)

    @staticmethod
    def from_id(snapshot_id: str, *, client: Client | None = None) -> "Snapshot":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            snapshot = resolved_client._bridge.run(
                NativeSnapshot.from_id(snapshot_id, client=resolved_client._client)
            )
        except BaseException:
            if owns_client:
                resolved_client.close()
            raise
        wrapped = resolved_client._wrap_snapshot(snapshot)
        wrapped._owns_client = owns_client
        return wrapped

    @staticmethod
    async def from_id_async(
        snapshot_id: str, *, client: Client | None = None
    ) -> "Snapshot":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            snapshot = await resolved_client._bridge.run_async(
                NativeSnapshot.from_id(snapshot_id, client=resolved_client._client)
            )
        except BaseException:
            if owns_client:
                await resolved_client.close_async()
            raise
        wrapped = resolved_client._wrap_snapshot(snapshot)
        wrapped._owns_client = owns_client
        return wrapped

    @property
    def id(self) -> str:
        return self._snapshot.id

    @property
    def name(self) -> str:
        return self._snapshot.name

    @property
    def status(self) -> SnapshotStatus:
        return self._snapshot.status

    @property
    def source_sandbox_id(self) -> str:
        return self._snapshot.source_sandbox_id

    @property
    def image_id(self) -> str | None:
        """The image this snapshot is pinned to, which a restore attaches.

        A restore always runs on this image, inferred by the backend;
        ``Sandbox.create`` rejects any ``image`` passed with ``snapshot``.
        """
        return self._snapshot.image_id

    @property
    def info(self) -> SnapshotInfo:
        return self._snapshot.info

    def refresh(self) -> "Snapshot":
        self._client._bridge.run(self._snapshot.refresh())
        return self

    async def refresh_async(self) -> "Snapshot":
        await self._client._bridge.run_async(self._snapshot.refresh())
        return self

    def wait_until_ready(self, *, timeout: float | None = None) -> "Snapshot":
        """Wait until capture finishes; by default, for as long as it takes.

        Holds the API's snapshot wait open where it is offered and polls where
        it is not, remembering which per client. Raises
        ``SnapshotFailedError`` when capture fails and ``SnapshotTimeoutError``
        when ``timeout`` seconds pass first. Neither a timeout nor
        cancellation deletes the snapshot.
        """
        self._client._bridge.run(self._snapshot.wait_until_ready(timeout=timeout))
        return self

    async def wait_until_ready_async(
        self, *, timeout: float | None = None
    ) -> "Snapshot":
        await self._client._bridge.run_async(
            self._snapshot.wait_until_ready(timeout=timeout)
        )
        return self

    def delete(self) -> None:
        """Delete the snapshot in any state, cancelling a capture in progress."""
        try:
            self._client._bridge.run(self._snapshot.delete())
        finally:
            if self._owns_client:
                self._client.close()

    async def delete_async(self) -> None:
        try:
            await self._client._bridge.run_async(self._snapshot.delete())
        finally:
            if self._owns_client:
                await self._client.close_async()


__all__ = ["Snapshot"]
