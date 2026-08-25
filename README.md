# Thunder Sandbox Python Client

Python-native access to Thunder sandboxes using the same credentials and local
state as the `tnr` CLI.

```python
import thunder

sandbox = thunder.Sandbox.create(cpu=4, memory=32, storage=50)
sandbox.wait_until_running()

process = sandbox.exec("uname", "-a")
print(process.stdout.read())
process.wait()

sandbox.upload("model.py", "/home/ubuntu/model.py")
sandbox.download("/home/ubuntu/results.json", "results.json")
sandbox.terminate()
```

The client reads `~/.thunder/cli_config.json` and honors the CLI overrides
`TNR_HOME`, `TNR_API_TOKEN`, and `TNR_API_URL`. It generates an Ed25519 key pair
before sandbox creation and stores it after creation as
`~/.thunder/sandbox_keys/<sandbox-name>` and `<sandbox-name>.pub`.

Existing key material can be supplied explicitly. Both halves are required
because sandbox SSH authorization is immutable:

```python
sandbox = thunder.Sandbox.create(
    ssh_public_key=public_key_text,
    ssh_private_key=private_key_text,
)
```

Commands use the system `ssh` executable and transfers use `scp`. A missing
private key is an error; the client never attempts to add or replace a key on
an existing sandbox.
