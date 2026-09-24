# Thunder Sandbox for Python

Create short-lived GPU sandboxes, run commands over SSH, and move files with a small, typed Python API.

Thunder Sandbox uses the same account and credentials as the Thunder CLI. It handles sandbox lifecycle, SSH key creation, waiting for readiness, command execution, uploads, and downloads through one synchronous and asynchronous Python API.

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

Alternatively, set `TNR_API_TOKEN` and, when using a non-default API endpoint, `TNR_API_URL`. API endpoints must use HTTPS.

## Quick start

```python
import thunder_sandbox as thunder

with thunder.Sandbox.create(
    cpu=4,
    memory=32,
    storage=50,
    gpu_type=thunder.GPUType.A6000,
    gpu_count=1,
    timeout=900,
).ephemeral() as sandbox:
    process = sandbox.exec("nvidia-smi", durable=True)
    output = process.stdout.read()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(process.stderr.read())
    print(output)
```

`wait_until_ready()` asks Thunder to hold the request open until the sandbox is ready, so it returns moments after startup finishes rather than on the next poll. Its `timeout` is enforced by the local SDK: each held request is bounded here and retried until the sandbox is ready, fails, or the timeout passes. An API without this endpoint is polled instead.

`ephemeral()` takes ownership of the sandbox, waits until it is ready, and terminates it on normal exit, exceptions, or readiness failure. Native async scopes also wait for bounded cleanup before propagating task cancellation. Use `ready_timeout` (default 300 seconds) and `cleanup_timeout` (default 120 seconds) to bound those operations. Keep a server-enforced sandbox TTL as a backstop for a killed Python process or unreachable API. Cleanup sends `/stop` even for a sandbox still in `CREATED`. If an older API returns `sandbox_not_ready`, the SDK retries that stop within the cleanup deadline; it does not start another readiness wait. If the API keeps refusing the stop, cleanup reports failure and the remote TTL remains the backstop. Cleanup failures are raised; when a scope was already failing, that original exception remains primary and the cleanup failure is chained as its cause. A failed cleanup is not a guarantee of remote termination. Do not enter this scope for a sandbox you intend to keep.

Observe readiness transitions with `sandbox.wait_until_ready(on_status=callback)` or `.ephemeral(on_status=callback)`. The callback receives `SandboxInfo` initially and when status or failure information changes. It is synchronous and must return promptly; with blocking API adapters it executes on the SDK event-loop thread, so do not call blocking SDK methods from it. The current API exposes sandbox lifecycle states, not detailed VM/image/container startup phases. Image resolution happens during `create()`, before these readiness notifications.

Sandboxes are addressed by `id`. SSH access uses an organization credential and a short-lived certificate, renewed by the SDK. Credentials are cached under `~/.thunder/sandbox_keys/`; reconnecting does not require a private key unique to the original sandbox. A `name` is an optional label. It must be free of any other *live* sandbox in the organization, and it is released once a sandbox finishes, so the same label can be reused later. A name never addresses a sandbox:

```python
sandbox = thunder.Sandbox.create(name="training-run", gpu_type=thunder.GPUType.H100)
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

Conditions worth retrying are typed, so they can be caught without matching on message text. Each carries the API's `code`, the HTTP `status`, and the server's `retry_after` hint when one was sent:

```python
try:
    sandbox = thunder.Sandbox.create(gpu_type=thunder.GPUType.H100)
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

Choose command behavior explicitly with `durable`:

| Options | Behavior | Stdin | Recoverable after disconnect |
|---|---|---|---|
| `durable=True` | Detached job with stored output | Unavailable | Yes, while job artifacts remain |
| `durable=False` | Attached process, separate stdout/stderr | Supported | No |
| `durable=False, pty=True` | Attached terminal, terminal output semantics | Supported | No |

For compatibility, omitting `durable` preserves the previous behavior: durable without a PTY, attached with a PTY. `durable=True, pty=True` is rejected. Output discard and `retain` apply only to durable jobs.

Attached execution supports ordinary stdin without allocating a terminal:

```python
process = sandbox.exec("cat", durable=False, text=False)
process.stdin.write(b"hello\n")
process.stdin.close()
print(process.stdout.read())
assert process.wait() == 0
```

Attached commands are never replayed: their input and execution state cannot be reconstructed safely after SSH disconnects. A connection error leaves their final remote state unknown. Use `durable=True` for unattended work that must survive reconnects.

Durable jobs store their status and output independently of the submission channel. Recover the same job without relaunching it:

```python
process = sandbox.exec("python3", "train.py", durable=True, retain=True)
process_id = process.id
recovered = sandbox.get_process(process_id)
exit_code = recovered.wait()
```

Durable stdin operations raise `io.UnsupportedOperation`. Use uploaded files, arguments, environment variables, or attached execution. `process.is_durable` reports the selected mode. Durable stdout and stderr are reconnectable streams. The SDK reads short SFTP chunks by byte offset and advances its cursor only after a complete chunk is in local memory, so a lost SSH connection resumes without gaps or duplicates. Text mode also preserves UTF-8 characters split across chunks.

Output is captured by default. Long-running commands which do not need one or both streams can redirect them directly to `/dev/null` in the durable launcher:

```python
process = sandbox.exec(
    "python3", "train.py", stdout="discard", stderr="capture"
)
```

Once a job is terminal and both captured streams have reached EOF, its remote job directory is removed automatically. Pass `retain=True` to keep it available for later `get_process()` recovery, and call `process.cleanup()` when finished. Explicit cleanup is idempotent and waits for a running job to finish.

`process.terminate()` for a durable job is also reconnectable. It records termination intent, signals the entire remote process group, and reconnects safely if the SSH acknowledgement is lost. Jobs which do not exit after a five-second `SIGTERM` grace period are stopped with `SIGKILL`; `wait()` then returns `143` or `137`.

Commands can set a working directory, environment variables, a timeout, or a pseudo-terminal:

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

Transfers intentionally restart from the beginning rather than maintaining a resumable byte manifest. Uploads first write to an isolated remote staging path; downloads first write beside the local destination. If SSH disconnects, the SDK reconnects and repeats the complete staged transfer. Completed files are published with an atomic rename, so partial data is never presented as the destination. Directory merges begin only after the network transfer completes.

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
    outbound_domain_allowlist=["pypi.org", "files.pythonhosted.org"],
)

# Block all outbound network access.
restricted.update_network_policy(block_network=True)

# Restore unrestricted outbound access.
restricted.update_network_policy()
```

`update_network_policy()` returns after Thunder accepts the desired policy; enforcement on the sandbox's node converges asynchronously. Tightening a policy blocks new connections but does not currently guarantee that already-established connections are terminated.

After the policy is accepted, the SDK attempts to flush DNS caches using `resolvectl flush-caches`: inside image-backed containers first (retrying with non-interactive `sudo` if needed), then on the VM with non-interactive `sudo` for all sandbox types. Nonzero flush exit codes are ignored; SSH transport errors still follow the normal retry and error behavior.

## Environment and lifetime

```python
sandbox = thunder.Sandbox.create(
    env={"EXPERIMENT": "baseline"},
    timeout=3600,
)
```

`timeout` is the sandbox lifetime in seconds. Set it to `None` to create a sandbox without an enforced TTL.

## Work with existing sandboxes

```python
with thunder.Client.from_cli() as client:
    for sandbox in client.list_sandboxes():
        print(sandbox.id, sandbox.status.value)

    sandbox = client.get_sandbox("sbx-0123456789abcdef")
    sandbox.wait_until_ready(timeout=300)
    print(" ".join(sandbox.ssh_command))
```

The SDK can reconnect from another machine using the same organization’s API credentials; it obtains SSH credentials as needed.

## Async API

Every blocking operation has an awaitable `_async` twin on the same public class. This makes it possible to use one import and pass `Client`, `Sandbox`, and `Process` objects between synchronous and asynchronous application code:

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

## Images and services

An image defines an execution environment. Thunder overrides its `ENTRYPOINT` and `CMD` with a keepalive process; creating an image-backed sandbox does not start the image’s application. `exec()` runs inside the container. Without an image, commands run in the guest VM. Existing positional arguments to `Sandbox.create()` still start the first process after readiness.

```python
image = thunder.Image.from_dockerfile("./runner-image")
with thunder.Sandbox.create(image=image, timeout=900).ephemeral() as sandbox:
    service = sandbox.start_service(
        "python3", "-m", "http.server", "8080", "--bind", "127.0.0.1",
        port=8080,
        ready_timeout=30,
    )
    print(service.id)
```

`start_service()` launches a durable foreground command and returns its `Process` after the port accepts TCP connections. The command must keep running, and the port must belong to that service. This is TCP readiness, not an HTTP or application health check. The probe runs in the guest VM, whose network namespace is shared by the container, and does not require SSH port forwarding. Startup failure or timeout requests process termination and waits up to 30 seconds. If that wait expires, the stop continues under the sandbox’s ownership; the readiness exception remains primary, with the cleanup error or timeout available through `__cause__`. Late termination errors are logged. Closing the sandbox’s SDK connection cancels remaining stop observations, so keep the connection open to let them finish or terminate the sandbox. Captured logs remain recoverable using the process ID in the exception. Successful service logs are also retained: terminate the service and call `service.cleanup()` when no longer needed, or terminate its sandbox.

Read output incrementally with `for line in service.stdout:` (native async: `async for line in service.stdout:`). A whole-stream `.read()` waits for EOF and is unsuitable for a service that keeps running. Standard output buffering still applies; use unbuffered application output when needed.

## Tunnel to sandbox services

```python
with sandbox.tunnel(50051) as endpoint:
    remote_executor = f"grpc://{endpoint.address}"
    # Run Bazel using this remote executor while the context is open.
```

`tunnel(remote_port, local_port=0, timeout=30)` binds only to `127.0.0.1` on your machine and connects to `127.0.0.1:remote_port` in the sandbox. Zero selects an available local port. The API exposes one sandbox service locally; it does not accept arbitrary destination hosts, bind publicly, create a remote listener, or expose the sandbox’s whole network. Start the service first: opening the tunnel checks that it accepts a TCP connection. The returned `Tunnel` exposes `host`, `port`, `address`, and `closed`, and supports explicit `close()` as well as context management.

The tunnel uses standard SSH TCP channels on a dedicated authenticated connection, without `socat`, shell relays, or a public service port. Closing a tunnel closes its accepted connections too, without disconnecting other sandbox commands. Terminating the sandbox or closing its SDK connection closes its tunnels. An interrupted tunnel does not reconnect or replay application traffic; create a new tunnel and let the application reconnect.

**Infrastructure prerequisite:** the sandbox SSH certificate must grant `permit-port-forwarding`; guest sshd must allow local TCP forwarding, restricted to loopback destinations (for example `AllowTcpForwarding local` and `PermitOpen 127.0.0.1:*`). Older deployments that disallow forwarding raise `UnsupportedFeatureError` with `code="ssh_forwarding_disabled"`. This SDK change does not modify deployed certificate authorities or VM images.

The native async API uses `async with await sandbox.tunnel(50051) as endpoint:`. Blocking objects support `async with await sandbox.tunnel_async(50051) as endpoint:`.

## Native async usage

For new asyncio applications, use `thunder_sandbox.asynchronous` directly; its operations run on your event loop. The top-level API and its `_async` adapters remain compatible for existing code.

```python
import asyncio
from thunder_sandbox.asynchronous import Sandbox

async def main():
    sandbox = await Sandbox.create(timeout=900)
    async with sandbox.ephemeral() as ready:
        process = await ready.exec("cat", durable=False)
        process.stdin.write("hello\n")
        await process.stdin.drain()
        process.stdin.write_eof()
        print(await process.stdout.read())
        assert await process.wait() == 0

asyncio.run(main())
```

Top-level objects provide the equivalent `async with sandbox.ephemeral_async():`, `start_service_async()`, and `tunnel_async()` methods. No existing methods were renamed or removed.

## Timeout meanings

| Setting | Meaning |
|---|---|
| `Sandbox.create(timeout=...)` | Remote sandbox lifetime/TTL; image resolution happens before allocation |
| `wait_until_ready(timeout=...)` / `ephemeral(ready_timeout=...)` | How long to wait for sandbox readiness |
| `exec(timeout=...)` / `Process.wait(timeout=...)` | How long to wait for command completion; expiry does not kill the command |
| `start_service(ready_timeout=...)` | How long to wait for the service port after command submission; failure attempts to terminate the process |
| `ephemeral(cleanup_timeout=...)` | Bound on scope-exit termination |
| `tunnel(timeout=...)` | Bound on establishing and probing the tunnel, not its lifetime |

## Configuration

Configuration is resolved from the following sources:

1. Explicit `ClientConfig` values.
2. `TNR_API_TOKEN` and `TNR_API_URL` environment variables.
3. Thunder CLI state in `~/.thunder/cli_config.json`.
4. The default Thunder API endpoint.

Set `TNR_HOME` to use a different directory for CLI state and sandbox SSH keys.

## License

Thunder Sandbox is available under the [Apache License 2.0](LICENSE).
