"""Exceptions raised by the Thunder sandbox client."""

from __future__ import annotations


class ThunderError(Exception):
    """Base class for all client errors.

    ``code`` is the API's machine-readable error identifier (for example
    ``sandbox_capacity_unavailable``) when one was returned, and ``status`` is
    the HTTP status. Both are exposed so callers can branch on the condition
    without matching on message text, which is not a stable contract.
    """

    def __init__(self, message: str, *, code: str | None = None,
                 status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after


class AuthenticationError(ThunderError):
    """The API token is missing, invalid, or expired."""


class ConnectionError(ThunderError):
    """The client could not communicate with Thunder."""


class RetryableError(ThunderError):
    """A transient condition; the same request may succeed if retried.

    ``retry_after`` carries the server's Retry-After hint in seconds when one
    was sent.
    """


class CapacityError(RetryableError):
    """No capacity is available for the requested resources right now.

    This is an ordinary scheduling outcome rather than a failure: the request
    was well formed and nothing is broken, there is simply no free GPU of the
    requested type. Retrying later is the correct response.
    """


class RateLimitError(RetryableError):
    """The client exceeded the API's rate limit for this route."""


class ServiceUnavailableError(RetryableError):
    """Thunder could not service the request and it was not applied."""


class InvalidRequestError(ThunderError, ValueError):
    """A request is invalid."""


class NotFoundError(ThunderError):
    """The requested object does not exist."""


class ConflictError(ThunderError):
    """The request conflicts with existing state."""


class UnsupportedFeatureError(ThunderError):
    """The requested feature is not supported by Thunder sandboxes."""


class SandboxError(ThunderError):
    """Base class for sandbox lifecycle errors."""


class SandboxFailedError(SandboxError):
    """Sandbox creation or execution failed."""


class SandboxTimeoutError(SandboxError, TimeoutError):
    """A sandbox operation exceeded its timeout."""
