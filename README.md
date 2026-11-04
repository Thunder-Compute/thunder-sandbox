# Thunder Sandbox for Python

Create short-lived GPU sandboxes, run commands over SSH, and move files with a
small, typed Python API.

Thunder Sandbox uses the same account and credentials as the Thunder CLI. It
handles sandbox lifecycle, SSH key creation, waiting for readiness, command
execution, uploads, and downloads through one synchronous and asynchronous
Python API.

## Installation

Thunder Sandbox requires Python 3.10 or newer.

```bash
pip install thunder-sandbox
```

To install the current development branch directly from GitHub:

```bash
pip install git+https://github.com/Thunder-Compute/thunder-sandbox.git
```

Authenticate with the Thunder CLI before using the library:

```bash
tnr login
```

Alternatively, set `TNR_API_TOKEN` and, when using a non-default API endpoint,
`TNR_API_URL`. API endpoints must use HTTPS.

## Quick start

```python
import thunder_sandbox as thunder

sandbox = thunder.Sandbox.create(
    cpu=4,
    memory=32,
    storage=50,
    gpu_type=thunder.GPUType.A6000,
    gpu_count=1,
    timeout=900,
)

try:
    sandbox.wait_until_ready()

    process = sandbox.exec("nvidia-smi")
    stdout = process.stdout.read()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(process.stderr.read())
    print(stdout)
finally:
    sandbox.terminate()
```

`wait_until_ready()` asks Thunder to hold the request open until the sandbox
is ready, so it returns moments after startup finishes rather than on the next
poll. Its `timeout` is enforced on the client: each held request is bounded
here and retried until the sandbox is ready, fails, or the timeout passes. An
API without this endpoint is polled instead.

Sandboxes are addressed by `id`. SSH connections use an Ed25519 key and a renewable organization certificate shared across the organization's sandboxes. The SDK creates credentials and renews certificates as needed, caching them under `~/.thunder/sandbox_keys/` when writable.

A `name` is an optional label. It must be free of any other *live* sandbox in
the organization, and it is released once a sandbox finishes, so the same label
can be reused later. A name never addresses a sandbox:

```python
sandbox = thunder.Sandbox.create(
    name="training-run", gpu_type=thunder.GPUType.H100, gpu_count=1
)
print(sandbox.id, sandbox.name)

# Claiming a name a live sandbox already holds raises ConflictError.
# Looking one up searches live sandboxes; prefer Sandbox.from_id.
same = thunder.Sandbox.from_name("training-run")
```

## Live pricing

Fetch current resource rates as a `thunder.Pricing` object, or get a sandbox's aggregate hourly rate:

```python
with thunder.Client.from_cli() as client:
    pricing = client.get_pricing()
    print(pricing.vcpu, pricing.memory_gb, pricing.storage_gb)
    print(pricing.gpu[thunder.GPUType.H100])

    sandbox = client.get_sandbox("sbx-your-id")
    print(sandbox.get_hourly_price())
```

All rates are USD per resource-hour. `memory_gb` and `storage_gb` are per **GiB**, matching sandbox resource quantities. The GPU dictionary contains only published rates, keyed by `GPUType`; A100XL is distinct from A100.

Each call fetches pricing from the API without SDK caching; the API's own caching still applies. The sandbox total includes every configured vCPU, GiB of memory, GiB of storage, and GPU, with no included-resource allowance or rounding. It uses the sandbox object's stored resources regardless of lifecycle status and reports an hourly rate, not accrued charges.

Invalid or missing base rates raise `ThunderError` with code `invalid_pricing`. A sandbox using an unpriced GPU raises `ThunderError` with code `pricing_unavailable` rather than returning a partial total.

The native asynchronous API supports `await client.get_pricing()` and `await sandbox.get_hourly_price()`. Blocking objects also expose `await client.get_pricing_async()` and `await sandbox.get_hourly_price_async()`.

## Handling errors

Conditions worth retrying are typed, so they can be caught without matching on
message text. Each carries the API's `code`, the HTTP `status`, and the
server's `retry_after` hint when one was sent:

```python
import time

try:
    sandbox = thunder.Sandbox.create(gpu_type=thunder.GPUType.H100, gpu_count=1)
except thunder.CapacityError as exc:
    # No free GPU of that type right now; the request was fine.
    time.sleep(exc.retry_after or 30)
except thunder.RetryableError:
    # Rate limited, or Thunder could not service the request.
    ...
```

## Run commands

Pass command arguments separately to avoid local shell interpretation:

```python
process = sandbox.exec("python3", "-c", "print('hello from Thunder')")
print(process.stdout.read())
exit_code = process.wait()
```

Non-PTY commands are launched as detached durable jobs. Their execution,
status, and output files do not depend on the SSH channel which submitted them.
If the connection is lost, the SDK reconnects and resumes observing the same
job rather than launching it again. Save `process.id` to recover it later:

```python
process_id = process.id
recovered = sandbox.get_process(process_id)
exit_code = recovered.wait()
```

Durable commands do not accept stdin; stdin operations raise
`io.UnsupportedOperation`. Pass input through arguments, environment variables,
or uploaded files. Commands created with `pty=True` retain interactive stdin but
remain attached to their SSH channel. `process.is_durable` reports which mode a
handle uses.

PTY commands are intentionally never replayed or recovered: input and terminal
state cannot be reconstructed safely after their SSH channel disappears. A PTY
disconnect raises `ConnectionError` and leaves the remote command's final state
unknown. Use the default `pty=False` mode for unattended or long-running work.

Durable stdout and stderr are reconnectable streams. After a lost connection, reading resumes without gaps or duplicate output. Text mode preserves UTF-8 characters across reads.

Output is captured by default. Use `"discard"` for streams you do not need:

```python
process = sandbox.exec(
    "python3", "train.py", stdout="discard", stderr="capture"
)
```

Once a command has finished and both captured streams have reached EOF, its saved state is removed automatically. Pass `retain=True` to keep it available for later `get_process()` recovery, and call `process.cleanup()` when finished. Explicit cleanup is idempotent and waits for a running command to finish.

`process.terminate()` is also durable. It records termination intent, signals
the entire remote process group, and reconnects safely if the SSH
acknowledgement is lost. Jobs which do not exit after a five-second `SIGTERM`
grace period are stopped with `SIGKILL`; `wait()` then returns `143` or `137`.

Commands can set a working directory, environment variables, a timeout, or a
pseudo-terminal:

```python
process = sandbox.exec(
    "python3",
    "train.py",
    workdir="/home/ubuntu/project",
    env={"MODEL": "llama", "DEBUG": "1"},
    timeout=600,
)
```

## Transfer files

```python
sandbox.upload("model.py", "/home/ubuntu/model.py")
sandbox.upload("dataset", "/home/ubuntu/dataset", recursive=True)

sandbox.download("/home/ubuntu/results.json", "results.json")
sandbox.download("/home/ubuntu/checkpoints", "checkpoints", recursive=True)
```

Transfers retry from the beginning after a lost connection. Files appear at their destination only after their contents have transferred successfully, so an interrupted transfer does not expose partially transferred files. Directory transfers merge with the destination after the network transfer completes.

## Network policies

Sandboxes have unrestricted outbound access by default. Restriction is always explicit:

```python
# No outbound internet access.
closed = thunder.Sandbox.create(block_network=True)

# Only the specified CIDRs and domains are permitted.
restricted = thunder.Sandbox.create(
    outbound_cidr_allowlist=["203.0.113.0/24"],
    outbound_domain_allowlist=["pypi.org", "files.pythonhosted.org"],
)
```

CIDR and domain allowlists are independent. Supply both when restricted workloads need both direct IP and DNS-based access.

For policy updates, `None` leaves that dimension unrestricted, while an empty sequence blocks it. Each call replaces the complete policy rather than merging with the previous allowlists.

Replace the complete outbound policy of a running sandbox with the same options used at creation:

```python
# Permit package downloads while blocking other destinations.
restricted.update_network_policy(
    outbound_cidr_allowlist=[],
    outbound_domain_allowlist=["pypi.org", "files.pythonhosted.org"],
)

# Block all outbound network access.
restricted.update_network_policy(block_network=True)

# Restore unrestricted outbound access.
restricted.update_network_policy()
```

`update_network_policy()` waits until Thunder confirms that the accepted policy generation is enforced. Both the blocking and asynchronous SDK paths wait for completion; asynchronous HTTP requests and polling yield to the event loop. The default timeout is 300 seconds; pass `timeout=...` to change it or `timeout=None` to wait indefinitely. Timeout or cancellation stops waiting but does not undo an accepted update. A superseded policy raises `ConflictError`, and a failed or stopped sandbox raises `SandboxFailedError`. Existing user connections are revoked during policy changes; SSH and permitted platform infrastructure remain available.

After a policy update is enforced, the SDK attempts to refresh DNS caches where supported.

Refreshed `sandbox.info` exposes `desired_network_policy_generation` and the last confirmed `network_policy_generation`.

## Environment and lifetime

```python
sandbox = thunder.Sandbox.create(
    env={"EXPERIMENT": "baseline"},
    timeout=3600,
)
```

`timeout` is the sandbox lifetime in seconds. Set it to `None` to create a
sandbox without an enforced TTL.

## Work with existing sandboxes

```python
with thunder.Client.from_cli() as client:
    for sandbox in client.list_sandboxes():
        print(sandbox.id, sandbox.status.value)

    sandbox = client.get_sandbox("sbx-0123456789abcdef")
    sandbox.wait_until_ready(timeout=300)
    print(" ".join(sandbox.ssh_command))
```

You can reconnect to existing sandboxes from another machine using API credentials authorized for that organization; the SDK obtains SSH credentials as needed. Running the printed `ssh_command` outside the SDK requires its cached private key and certificate.

## Async API

Every blocking operation has an awaitable `_async` twin on the same public
class. This makes it possible to use one import and pass `Client`, `Sandbox`,
`Snapshot`, and `Process` objects between synchronous and asynchronous
application code:

```python
import asyncio
import thunder_sandbox as thunder


async def main() -> None:
    sandbox = await thunder.Sandbox.create_async(
        gpu_type=thunder.GPUType.A6000,
        gpu_count=1,
    )
    try:
        await sandbox.wait_until_ready_async()
        process = await sandbox.exec_async("nvidia-smi")
        exit_code = await process.wait_async()
        if exit_code != 0:
            raise RuntimeError(await process.stderr.read_async())
        print(await process.stdout.read_async())
    finally:
        await sandbox.terminate_async()


asyncio.run(main())
```

## Container images

An image-backed sandbox runs the image's `ENTRYPOINT` followed by its `CMD` arguments. If only one is set, it runs that command. The initial command is attempted once. After it exits or fails, the sandbox stays alive and available for `exec` and file transfers until you terminate it or its lifetime expires. If the container needs to restart, Thunder restores that availability without relaunching the initial command.

For example, put this `Dockerfile` in `./server-image` to start a Python HTTP server automatically:

```dockerfile
FROM python:3.12-slim
ENTRYPOINT ["python", "-m", "http.server"]
CMD ["8080", "--bind", "0.0.0.0"]
```

```python
sandbox = thunder.Sandbox.create(
    image=thunder.Image.from_dockerfile("./server-image"),
    timeout=900,
)
try:
    sandbox.wait_until_ready()
    # Additional commands execute inside the running server container.
    process = sandbox.exec("python", "--version")
    print(process.stdout.read())
    process.wait()
finally:
    sandbox.terminate()
```

`Image.from_registry(...)` uses the same image-command behavior. Inherited defaults count too: a base image's shell or interpreter command can exit immediately when started without interactive input.

If both `ENTRYPOINT` and `CMD` are absent or empty, the sandbox stays available for `sandbox.exec(...)` without running an initial command. To prevent an inherited command from running, clear both defaults:

```dockerfile
FROM ubuntu:24.04
ENTRYPOINT []
CMD []
```

Positional arguments to `Sandbox.create("python", "script.py", image=image)` run as a separate `exec` command after sandbox readiness. They do not replace the image's `ENTRYPOINT` or `CMD`. The image workload may already be running or may have exited before this additional command starts. Without an image, commands run directly in the sandbox.

`wait_until_ready()` waits for sandbox initialization, not application health. It can succeed even if a short image command has already finished. Commands and file transfers remain available after that command exits. Call `terminate()` when finished, or use a finite sandbox lifetime to release its resources.

`sandbox.wait()` and `sandbox.poll()` track a positional command passed to `create`, or otherwise the sandbox's lifecycle. They do not track the image command's exit or return its exit code. The SDK does not expose a `Process` handle, captured output, or exit status for the image's initial command. Commands launched with `sandbox.exec(...)` retain their normal `Process` output and wait behavior. These semantics also apply to the async APIs.

Use `user` to choose the account for an individual command, without requiring `su`, `sudo`, or a password inside the image:

```python
setup = sandbox.exec("mkdir", "-p", "/opt/agent-tools", user="root")
setup.wait()
agent = sandbox.exec("id", user="1000:1000")
```

`user` accepts a username or UID, optionally followed by a group name or GID (`"root"`, `"1000"`, or `"1000:1000"`). Commands default to root unless `user` is specified, independently of the image's `USER`. This option works for durable and PTY commands through `exec` and `exec_async`, including the native async SDK. Specifying `user` without an image raises `InvalidRequestError`.

## Snapshots

A snapshot is a durable copy of a ready sandbox's disk. Restoring it creates a new sandbox whose files match the source at capture time. It outlives the source sandbox, so you can terminate the source once the snapshot is ready:

```python
with thunder.Client.from_cli() as client:
    source = client.create_sandbox(image=thunder.Image.from_registry("python:3.12"))
    source.wait_until_ready()
    source.exec("sh", "-c", "pip install numpy > /root/setup.log").wait()

    snapshot = source.snapshot(name="numpy-ready")
    snapshot.wait_until_ready(timeout=1800)
    source.terminate()

    restored = client.create_sandbox(
        snapshot=snapshot, gpu_type=thunder.GPUType.H100, gpu_count=1
    )
    try:
        restored.wait_until_ready()
        process = restored.exec("cat", "/root/setup.log")
        process.wait()
        print(restored.snapshot_id, process.stdout.read())
    finally:
        restored.terminate()
```

Snapshots capture disk state only. They do not capture memory, running processes, open connections, or GPU state. A restored sandbox starts fresh from that disk: its image command runs again, and `wait_until_ready()` behaves as it does for any new sandbox.

`create_sandbox(snapshot=...)` accepts a `Snapshot` or its ID and cannot be combined with `image`, which raises `InvalidRequestError` before any request. The restored sandbox uses the source's image. Every other option applies as usual, including a different GPU type or count. A `storage` value smaller than the snapshot is grown to fit it. Restoring a snapshot that is not yet ready raises `ConflictError`.

`wait_until_ready()` waits until capture finishes, holding the request open on the API where it offers a snapshot wait and polling otherwise. A failed capture raises `SnapshotFailedError`, whose `failure_code` and `failure` give the recorded reason. When `timeout` passes first, it raises `SandboxTimeoutError`. A timeout or cancellation does not delete the snapshot: capture continues server-side, and a later wait picks it up. A snapshot deleted while waiting raises `NotFoundError`.

A snapshot uses the client of the sandbox that created it. A sandbox created without an explicit client closes its own client on `terminate()`. In that case, wait on the snapshot before terminating the source, or use a shared `Client` as shown above.

```python
with thunder.Client.from_cli() as client:
    for snapshot in client.list_snapshots(status=thunder.SnapshotStatus.READY):
        print(snapshot.id, snapshot.name, snapshot.info.storage_gib)

    snapshot = client.get_snapshot("snap-your-id")
    snapshot.delete()
```

`delete()` works in any state and cancels a capture in progress. Creating a snapshot of a sandbox that is not ready raises `ConflictError`. Exceeding the organization's snapshot limit raises `ConflictError`; delete a snapshot before creating another. Every method has an `_async` twin, and the native asynchronous API offers the same operations on `thunder_sandbox.asynchronous.Snapshot`.

## Configuration

Configuration is resolved from the following sources:

1. Explicit `ClientConfig` values.
2. `TNR_API_TOKEN` and `TNR_API_URL` environment variables.
3. Thunder CLI state in `~/.thunder/cli_config.json`.
4. The default Thunder API endpoint.

Set `TNR_HOME` to use a different directory for CLI state and sandbox SSH keys.

## License

Thunder Sandbox is available under the [Apache License 2.0](LICENSE).
