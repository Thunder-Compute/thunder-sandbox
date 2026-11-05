"""Native asynchronous handle to a durable sandbox disk snapshot."""

from __future__ import annotations

import time
from dataclasses import replace

from .._common.exceptions import (
    ConnectionError,
    InvalidRequestError,
    RetryableError,
    SnapshotFailedError,
    SnapshotTimeoutError,
)
from .._common.types import SnapshotInfo, SnapshotStatus
from ._polling import (
    OUTAGE_GRACE_SECONDS,
    _datetime,
    _path_segment,
    _sleep_until_next_poll,
)
from .client import Client

# Capture takes minutes, so a longer cap than the sandbox waiter's keeps a long
# capture from spending the read rate limit; a few seconds of latency do not
# matter.
SNAPSHOT_POLL_MAX_SECONDS = 10.0


class Snapshot:
    """A native asynchronous handle to a durable sandbox disk snapshot.

    A snapshot holds the source sandbox's whole disk, including the
    container's changes to its image, but not the image itself, which it pins
    as :attr:`image_id`. Restoring it creates a new sandbox with those files,
    not the source's memory, processes, or GPU state. It outlives its source
    sandbox until it is deleted. A snapshot is not an image and cannot be
    passed as ``image=``.
    """

    def __init__(
        self, client: Client, info: SnapshotInfo, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._owns_client = owns_client
        self._info = info

    @staticmethod
    async def from_id(
        snapshot_id: str, *, client: Client | None = None
    ) -> "Snapshot":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            response = await resolved_client._request(
                "GET", _snapshot_path(snapshot_id)
            )
            snapshot = Snapshot._from_response(resolved_client, response)
            snapshot._owns_client = owns_client
            return snapshot
        except BaseException:
            if owns_client:
                await resolved_client.close()
            raise

    @staticmethod
    def _from_response(client: Client, response: dict[str, object]) -> "Snapshot":
        return Snapshot(client, _snapshot_from_response(response))

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def status(self) -> SnapshotStatus:
        return self._info.status

    @property
    def source_sandbox_id(self) -> str:
        return self._info.source_sandbox_id

    @property
    def image_id(self) -> str | None:
        """The image this snapshot is pinned to, which a restore attaches.

        A restore always runs on this image, inferred by the backend;
        ``Sandbox.create`` rejects any ``image`` passed with ``snapshot``.
        """
        return self._info.image_id

    @property
    def info(self) -> SnapshotInfo:
        return self._info

    async def refresh(self) -> "Snapshot":
        response = await self._client._request("GET", _snapshot_path(self.id))
        self._info = _snapshot_from_response(response)
        return self

    async def wait_until_ready(self, *, timeout: float | None = None) -> "Snapshot":
        """Wait until capture finishes; by default, for as long as it takes.

        Polls the snapshot with a capped backoff. Raises
        :class:`SnapshotFailedError` when capture fails and
        :class:`SnapshotTimeoutError` when ``timeout`` seconds pass first.
        Neither a timeout nor cancellation deletes the snapshot: capture
        continues server-side and a later wait can pick it up. A snapshot
        deleted while this waits surfaces as :class:`NotFoundError` from the
        next read.
        """
        if timeout is not None and not timeout >= 0:
            raise InvalidRequestError("snapshot timeout cannot be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        delay = 1.0
        while True:
            try:
                await self.refresh()
            except (ConnectionError, RetryableError) as exc:
                now = time.monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= OUTAGE_GRACE_SECONDS:
                    raise
                if deadline is not None and now >= deadline:
                    raise
                await _sleep_until_next_poll(deadline, delay, exc.retry_after)
                delay = min(SNAPSHOT_POLL_MAX_SECONDS, delay * 2.0)
                continue
            failing_since = None
            if self.status == SnapshotStatus.READY:
                return self
            if self.status == SnapshotStatus.FAILED:
                info = self._info
                prefix = f"[{info.failure_code}] " if info.failure_code else ""
                raise SnapshotFailedError(
                    f"snapshot {self.id} failed: {prefix}"
                    + (info.failure or "capture failed"),
                    code=info.failure_code,
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise SnapshotTimeoutError(
                    f"snapshot {self.id} did not become ready within {timeout} "
                    f"seconds (status: {self.status.value}); this wait did not "
                    "delete it"
                )
            # The sleep stops at the deadline and one last read follows it,
            # so readiness arriving during the pause is still seen.
            await _sleep_until_next_poll(deadline, delay)
            delay = min(SNAPSHOT_POLL_MAX_SECONDS, delay * 2.0)

    async def delete(self) -> None:
        """Delete the snapshot in any state, cancelling a capture in progress."""
        try:
            await self._client.delete_snapshot(self.id)
            self._info = replace(self._info, status=SnapshotStatus.DELETING)
        finally:
            if self._owns_client:
                await self._client.close()


def _snapshot_path(snapshot_id: str) -> str:
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise InvalidRequestError("snapshot ID must be a non-empty string")
    return f"/sandbox-snapshots/{_path_segment(snapshot_id)}"


def _optional_string(value: object) -> str | None:
    return str(value) if value else None


def _snapshot_from_response(response: dict[str, object]) -> SnapshotInfo:
    snapshot_id = response.get("id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ConnectionError("Thunder returned a snapshot without an ID")
    storage = response.get("storage_gib")
    return SnapshotInfo(
        id=snapshot_id,
        name=str(response.get("name") or ""),
        status=SnapshotStatus(str(response.get("status") or "")),
        source_sandbox_id=str(response.get("source_sandbox_id") or ""),
        storage_gib=storage if type(storage) is int else 0,
        created_at=_datetime(response.get("created_at")),
        image_id=_optional_string(response.get("image_id")),
        ready_at=_datetime(response.get("ready_at"), optional=True),
        expires_at=_datetime(response.get("expires_at"), optional=True),
        failure_code=_optional_string(response.get("failure_code")),
        failure=_optional_string(response.get("failure")),
    )


__all__ = ["Snapshot"]
