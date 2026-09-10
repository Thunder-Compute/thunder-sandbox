"""Deterministic fault injection shared by durable SSH tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping


class SSHDisconnected(OSError):
    """A synthetic transport loss raised at a named protocol checkpoint."""


class SSHFaults:
    """Raise scheduled failures at deterministic operation occurrences.

    The schedule maps a checkpoint name and its one-based occurrence to an
    exception factory. Factories provide a fresh exception for every failure,
    which keeps traceback state isolated when a plan is reused.
    """

    def __init__(
        self,
        schedule: Mapping[str, Mapping[int, Callable[[], BaseException]]] | None = None,
    ) -> None:
        self._schedule = {
            name: dict(occurrences) for name, occurrences in (schedule or {}).items()
        }
        self._hits: Counter[str] = Counter()

    async def checkpoint(self, name: str) -> None:
        self._hits[name] += 1
        factory = self._schedule.get(name, {}).get(self._hits[name])
        if factory is not None:
            raise factory()

    def hits(self, name: str) -> int:
        return self._hits[name]


class FakeSSHConnection:
    """The connection lifecycle surface needed by retry-manager tests."""

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    async def wait_closed(self) -> None:
        return None


def disconnect(message: str = "injected SSH disconnect") -> Callable[[], BaseException]:
    return lambda: SSHDisconnected(message)


__all__ = ["FakeSSHConnection", "SSHDisconnected", "SSHFaults", "disconnect"]
