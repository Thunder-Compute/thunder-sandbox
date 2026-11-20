"""Execution channels: the transports a sandbox's processes and files go through."""

from ._base import (
    SUPPORTED,
    ExecHandle,
    ExecutionChannel,
    Fixed,
    SupportTable,
    Unsupported,
    optional_fields,
)

__all__ = [
    "SUPPORTED",
    "ExecHandle",
    "ExecutionChannel",
    "Fixed",
    "SupportTable",
    "Unsupported",
    "optional_fields",
]
