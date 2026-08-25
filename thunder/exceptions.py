"""Exceptions raised by the Thunder sandbox client."""


class ThunderError(Exception):
    """Base class for all client errors."""


class AuthenticationError(ThunderError):
    """The API token is missing, invalid, or expired."""


class ConnectionError(ThunderError):
    """The client could not communicate with Thunder."""


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
