"""Create a sandbox, wait for readiness, and print its SSH command.

Run with credentials from the Thunder CLI, optionally overriding the API with
TNR_API_URL:

    bazel run //thunder-client/test:sandbox_smoke_test
"""

from __future__ import annotations

import argparse
import shlex

import thunder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--memory", type=int, default=32, help="Memory in GiB")
    parser.add_argument("--storage", type=int, default=50, help="Storage in GiB")
    parser.add_argument("--ttl", type=int, default=900, help="Sandbox lifetime in seconds")
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=600,
        help="Maximum seconds to wait for the sandbox to become ready",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sandbox = thunder.Sandbox.create(
        cpu=args.cpu,
        memory=args.memory,
        storage=args.storage,
        gpu="A6000",
        timeout=args.ttl,
    )
    print(f"Created sandbox {sandbox.name}; waiting for readiness...", flush=True)

    sandbox.wait_until_running(timeout=args.ready_timeout)

    print(f"Sandbox {sandbox.name} is ready.")
    print(shlex.join(sandbox.ssh_command))


if __name__ == "__main__":
    main()
