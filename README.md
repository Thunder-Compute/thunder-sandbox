# Thunder Sandbox for Python

Create short-lived GPU sandboxes, run commands over SSH, and move files with a
small, typed Python API.

Thunder Sandbox uses the same account and credentials as the Thunder CLI. It
handles sandbox lifecycle, SSH key creation, readiness polling, command
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
`TNR_API_URL`.

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

Sandboxes are addressed by `id`. `Sandbox.create()` generates an Ed25519 key
pair and stores it under `~/.thunder/sandbox_keys/<sandbox-id>`. The public key
is immutable for the lifetime of the sandbox.

A `name` is an optional label. It must be free of any other *live* sandbox in
the organization, and it is released once a sandbox finishes, so the same label
can be reused later. A name never addresses a sandbox:

```python
sandbox = thunder.Sandbox.create(name="training-run", gpu_type=thunder.GPUType.H100)
print(sandbox.id, sandbox.name)

# Claiming a name a live sandbox already holds raises ConflictError.
# Looking one up searches live sandboxes; prefer Sandbox.from_id.
same = thunder.Sandbox.from_name("training-run")
```

## Handling errors

Conditions worth retrying are typed, so they can be caught without matching on
message text. Each carries the API's `code`, the HTTP `status`, and the
server's `retry_after` hint when one was sent:

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

## Network policies

Sandboxes have unrestricted outbound access by default. Restriction is always
explicit:

```python
# No outbound internet access.
closed = thunder.Sandbox.create(block_network=True)

# Only the specified CIDRs and domains are permitted.
restricted = thunder.Sandbox.create(
    outbound_cidr_allowlist=["203.0.113.0/24"],
    outbound_domain_allowlist=["pypi.org", "files.pythonhosted.org"],
)
```

CIDR and domain allowlists are independent. Supply both when restricted
workloads need both direct IP and DNS-based access.

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

The private SSH key must still exist locally to execute commands or transfer
files against an existing sandbox.

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

## Configuration

Configuration is resolved from the following sources:

1. Explicit `ClientConfig` values.
2. `TNR_API_TOKEN` and `TNR_API_URL` environment variables.
3. Thunder CLI state in `~/.thunder/cli_config.json`.
4. The default Thunder API endpoint.

Set `TNR_HOME` to use a different directory for CLI state and sandbox SSH keys.

## License

Thunder Sandbox is available under the [Apache License 2.0](LICENSE).
