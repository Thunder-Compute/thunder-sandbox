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

## CPU-only sandboxes and queueing

Omit GPU options for a CPU-only sandbox, or explicitly pass `gpu_type=thunder.GPUType.NONE, gpu_count=0`. CPU, memory and storage are whole numbers, with memory and storage in GiB. The API's resource catalog sets the allowed range for each and rejects a value outside it with `InvalidRequestError`, so small sizes such as `memory=1` may not be available.

```python
sandbox = thunder.Sandbox.create(
    cpu=4,
    memory=16,
    storage=20,
    gpu_type=thunder.GPUType.NONE,
    gpu_count=0,
    queue=True,
    timeout=900,
)
try:
    print(sandbox.queued)  # Last observed waiting-for-capacity state.
    sandbox.wait_until_ready(timeout=None)
finally:
    sandbox.terminate()
```

`queue=True` admits image builds/imports without checking future sandbox capacity and lets the sandbox wait for capacity without a scheduling timeout. When creating with `image=...`, the SDK passes this option to both image preparation and sandbox creation. Standalone preparation also accepts `client.resolve_image(image, queue=True)`. Validation, account limits, and infrastructure errors still apply; image builds still require Kubernetes workers. Omitting the flag preserves normal admission behavior.

Creation without positional commands returns a handle while the sandbox is queued, after any image preparation finishes. Refresh the handle with `sandbox.refresh()` to update `sandbox.queued` and `sandbox.info.queued`. `terminate()` cancels a queued sandbox. The sandbox's runtime TTL (`timeout` on creation) starts at readiness for queue-admitted sandboxes; `wait_until_ready(timeout=None)` waits indefinitely. Positional commands still wait for readiness within the creation `timeout`, so use a handle followed by an explicit wait for long queue waits. Image preparation retains its own timeout (two hours by default). The same options work with the native async SDK and the sync SDK's async adapters.

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

Sandboxes are addressed by `id`. SSH connections use an Ed25519 key and a renewable certificate shared across a workspace's sandboxes. The SDK creates credentials and renews certificates as needed, caching them under `~/.thunder/sandbox_keys/` when writable.

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

## Workspaces

Workspaces divide an organization's sandboxes, and each member has a role in each workspace. Choose one by name or by its `ws-` ID:

```python
with thunder.Client.from_cli() as client:
    for workspace in client.list_workspaces():
        print(workspace.id, workspace.name, workspace.role)

    sandbox = client.create_sandbox(workspace="research", cpu=4, memory=16)
    print(sandbox.workspace_id)

    for sandbox in client.list_sandboxes(workspace="research"):
        print(sandbox.id)

    sandbox = client.get_sandbox_by_name("worker", workspace="research")
```

Clients are not tied to a workspace. Each call uses the first of:

1. The `workspace` argument.
2. The `TNR_WORKSPACE` environment variable.
3. `main`, which every organization has. Code that never mentions a workspace uses it.

This applies to creating sandboxes, listing them and finding them by name. A sandbox ID needs no workspace, so `client.get_sandbox(id)` finds a sandbox in any workspace you can access, and everything you then do with it follows that sandbox.

Images are built or imported in a workspace too: `Sandbox.create(image=...)` uses the workspace the sandbox starts in, and `client.resolve_image(image, workspace="research")` prepares one ahead of time. The organization builds each recipe once, so a Dockerfile or registry digest already built in another workspace comes back as the same image, ready at once, and is then listed in yours as well. Starting a sandbox from an image ID that another workspace lists, as the REST API allows, takes the developer role in a workspace that lists it and adds the image to the sandbox's workspace; otherwise the start fails with `NotFoundError`.

Viewers can read a workspace's sandboxes. Starting, stopping and connecting to them, and building or importing images there, takes the developer role.

`list_sandboxes` covers one workspace. Names can change, so anything long-lived, such as a cleanup job, should store workspace IDs.

## Live pricing

Fetch current resource rates as a `thunder.Pricing` object, or get a sandbox's aggregate rate per second:

```python
with thunder.Client.from_cli() as client:
    pricing = client.get_pricing()
    print(pricing.vcpu, pricing.memory_gb, pricing.storage_gb)
    print(pricing.gpu[thunder.GPUType.H100])

    sandbox = client.get_sandbox("sbx-your-id")
    print(sandbox.get_price_per_second())
```

`Pricing` rates are USD per resource-hour. `memory_gb` and `storage_gb` are per **GiB**, matching sandbox resource quantities. The GPU dictionary contains only published rates, keyed by `GPUType`; A100XL is distinct from A100.

Each call fetches pricing from the API without SDK caching; the API's own caching still applies. The sandbox total includes every configured vCPU, GiB of memory, GiB of storage, and GPU, with no included-resource allowance or rounding. It uses the sandbox object's stored resources regardless of lifecycle status and reports USD per second (the hourly rates divided by 3600), not accrued charges.

Invalid or missing base rates raise `ThunderError` with code `invalid_pricing`. A sandbox using an unpriced GPU raises `ThunderError` with code `pricing_unavailable` rather than returning a partial total.

The native asynchronous API supports `await client.get_pricing()` and `await sandbox.get_price_per_second()`. Blocking objects also expose `await client.get_pricing_async()` and `await sandbox.get_price_per_second_async()`.

## Sandbox lifecycle timings

`sandbox.info` carries the sandbox's lifecycle timestamps, all timezone-aware: `created_at`, `ready_at`, `stopped_at` (finished or failed) and `expires_at` (when the TTL ends it). Sandboxes from `list_sandboxes`, `get_sandbox` and `create` carry the same fields, so reading them never makes a request:

```python
with thunder.Client.from_cli() as client:
    for sandbox in client.list_sandboxes(status="finished"):
        print(sandbox.id, sandbox.info.ready_at, sandbox.info.stopped_at)
```

A timestamp is `None` until its transition happens: `ready_at` before the sandbox is ready, or for good if it failed first, and `stopped_at` while it runs. The values are as of the last fetch, so call `sandbox.refresh()` to see transitions since then.

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

In an image-backed sandbox, a command given by bare name runs the image's own program, as it would under `docker exec`. So `sh` is the image's shell, which is dash on Ubuntu and Debian and doesn't support `[[ ]]`, `set -o pipefail` or `source`; use `bash` for scripts that need them. If the image has no such program, the sandbox's BusyBox provides it when it can.

`wait()` returns when the command itself exits, so a command can leave a background service running and return immediately. In an image-backed sandbox, redirect the service's output if you start it with `&`; otherwise it holds the container's output stream open and `wait()` lasts as long as the service:

```python
sandbox.exec("sh", "-c", "my-server >/var/log/my-server.log 2>&1 </dev/null &").wait()
```

Each argument or environment value must be under 128 KiB, the Linux limit; `exec` raises `InvalidRequestError` above that. Upload larger input as a file.

If the connection fails, the SDK reconnects and retries. An operation which keeps failing for ten minutes raises `ConnectionError`; a durable command keeps running, and `get_process()` can recover it.

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

To make creation safe to retry, give the sandbox a unique `name`. Names are unique among live sandboxes, so if a create request is interrupted and you can't tell whether it succeeded, `Sandbox.from_name(name)` finds the sandbox if it exists:

```python
try:
    sandbox = thunder.Sandbox.create(name=run_name, timeout=900)
except thunder.ConnectionError:
    sandbox = thunder.Sandbox.from_name(run_name)
```

## Transfer files

```python
sandbox.upload("model.py", "/home/ubuntu/model.py")
sandbox.upload("dataset", "/home/ubuntu/dataset", recursive=True)

sandbox.download("/home/ubuntu/results.json", "results.json")
sandbox.download("/home/ubuntu/checkpoints", "checkpoints", recursive=True)
```

Pass `timeout=` to bound a transfer in seconds; it raises `SandboxTimeoutError` when it expires. Transfers retry from the beginning after a lost connection. Files appear at their destination only after their contents have transferred successfully, so an interrupted transfer does not expose partially transferred files. Directory transfers merge with the destination after the network transfer completes.

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

Passing either allowlist restricts outbound traffic to what you list; an allowlist you leave out allows nothing. Supply both when restricted workloads need both direct IP and DNS-based access. To allow everything through one dimension, say so explicitly with `outbound_cidr_allowlist=["0.0.0.0/0"]` or `outbound_domain_allowlist=["*"]`. Passing neither leaves the sandbox open.

Domain allowlists cover TLS connections only, matched on the server name. Plain HTTP to an allowed domain fails, so allow the destination's CIDR as well for tools that need it, such as `apt` with `http://` mirrors.

Each call replaces the complete policy rather than merging with the previous allowlists.

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
and `Process` objects between synchronous and asynchronous application code:

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

When the Dockerfile's `FROM` image is in a private registry, pass a credential for that registry:

```python
image = thunder.Image.from_dockerfile(
    "./server-image",
    registry_host="ghcr.io",
    registry_username="my-user",
    registry_password=os.environ["GHCR_TOKEN"],
)
```

The credential is offered only to `registry_host` (use `docker.io` for Docker Hub). Base images from any other registry must be public. Credentials are not part of the image's cache key: an already-built image with the same build context is reused without pulling again.

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

## Configuration

Configuration is resolved from the following sources:

1. Explicit `ClientConfig` values.
2. `TNR_API_TOKEN` and `TNR_API_URL` environment variables.
3. Thunder CLI state in `~/.thunder/cli_config.json`.
4. The default Thunder API endpoint.

Set `TNR_HOME` to use a different directory for CLI state and sandbox SSH keys.

Set `TNR_WORKSPACE` to choose the workspace for calls that do not name one. See [Workspaces](#workspaces).

## License

Thunder Sandbox is available under the [Apache License 2.0](LICENSE).
