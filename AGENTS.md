# Cleanup backlog: `rohan/ssh_reliability`

Findings from a code-quality audit of `c90aa74..dfca182` (the durable-SSH-job branch: 14 files, +3610/-145). The durable-job *design* is sound — atomic-claim launcher, offset-cursor reconnectable readers, staged-then-renamed transfers. Every item below is about the *implementation* left around it. **All of it is behavior-preserving.** Nothing here should change what the SDK does.

Work top-down; items 1-7 are blockers. Line numbers are as of `dfca182` and will shift as you edit — re-locate by symbol name, not by line.

## Ground rules

- No behavior changes. If a cleanup seems to require one, stop and flag it instead.
- Tests must pass after every item: `.venv/bin/python -m unittest discover -s test -p "*_test.py" -t .` (127 tests, ~0.5s). `asyncssh` is not installed in the system Python — use `.venv/bin/python`.
- CI matrix is Python 3.10-3.14 on Linux plus 3.14 on macOS/Windows. Do not use 3.11+ syntax.
- Prefer deleting complexity over rearranging it. If a fix makes a file bigger, it is probably the wrong fix.

---

## Blockers

### 1. Decompose `asynchronous/sandbox.py` (1042 -> 1798 lines)

One module now holds sandbox lifecycle, SSH execution, a durable-job protocol client, a file-transfer engine, local filesystem publish helpers, connection management, and network policy. Split along the seams that already exist:

- `_launch_detached_job`, `_durable_process`, `_read_job_status`, `_read_job_spec`, `_read_job_output`, `_cleanup_job`, `_signal_job` (`sandbox.py:412-776`, ~360 lines) -> new **`_jobclient.py`**. These only speak the `_jobs.py` protocol over an `SSHConnectionManager`; they touch nothing else on `Sandbox`.
- `upload`, `download`, and the nine `_*_transfer*` / `_upload_*` / `_download_*` helpers (`sandbox.py:784-1145`, ~360 lines) -> new **`_transfer.py`**.
- `_remove_local_transfer_path`, `_publish_local_transfer`, `_replace_local_transfer`, `_publish_local_contents` (`sandbox.py:1452-1501`) -> also `_transfer.py`. These are pure local-filesystem functions: no sandbox, no SSH, no async. They have no business in this module.

Target: `Sandbox` around 500 lines doing what its name says.

### 2. Split `Process` into two classes; delete the `__new__` construction

`process.py:316` builds the durable variant with `cls.__new__(cls)` and hand-assigns fifteen attributes, duplicating the whole field list from `__init__`. The damage is visible in the same file:

- Eight `if self._process is not None` mode branches at `process.py:367, 375, 378, 387, 459, 482, 488` across `returncode`, `poll`, `wait`, `cleanup`, `terminate`, `_maybe_auto_cleanup`.
- `__del__` (`process.py:536`) uses `getattr(self, "_process", None)` with defaults *because* a bypassed `__init__` means fields may not exist.
- `_maybe_auto_cleanup` casts its own attributes: `cast(_DurableReader[T], self.stdout)` (`process.py:462-463`).
- `Process.durable` is annotated `-> "Process[str] | Process[bytes]"` on a classmethod of `Process[T]` (a type lie), forcing another `cast` at the caller (`sandbox.py:526`).
- `stdout: str = "capture"` / `stderr: str = "capture"` (`process.py:328-329`) should be `OutputMode`.

Make `Process` an ABC holding the shared reader/writer surface, with `AttachedProcess` and `DurableProcess` subclasses. `Process` stays the exported public name (`asynchronous/__init__.py:6`, `synchronous/__init__.py:6`, `thunder_sandbox/__init__.py:6`) so nothing downstream changes. There are no `isinstance` checks against it anywhere. Every branch, cast, the `__new__`, and the defensive `getattr` all disappear.

### 3. Collapse six copies of the SSH-run boilerplate into the helper that already exists

`_run_idempotent_command` (`sandbox.py:1021`) already *is* "run a command through the retry manager, treat a lost acknowledgement as retryable, raise with stderr detail on nonzero." Then five methods re-implement it inline:

- 7 hand-written `RetryableSSHOperationError` closures: `sandbox.py:457, 539, 569, 633, 754, 931, 1029`
- 6 copies of `detail = str(result.stderr or "").strip()` plus its f-string error block
- 3 copies of `JobStatus.from_json(str(result.stdout or "").strip())` plus a near-identical try/except

The only real differences: submission and signalling also retry exit code 75, and each site wants a different exception type on failure. That is two keyword arguments:

```python
async def _run_idempotent_command(
    self, command: str, *, name: str, deadline: float | None = None,
    retry_exit_codes: Container[int] = (),
    failure: type[Exception] = SandboxFailedError,
    failure_message: str | None = None,
) -> str:  # returns stdout
```

Plus a three-line `_decode_status(text, context)` for the three parse sites. ~130 lines become ~50; `_read_job_status` becomes four lines. Highest-value single change in the backlog — do it before 1, since it shrinks what has to move.

### 4. One source of truth for the job status wire format

`_jobs.py:1-6` states the module "deliberately contains no SSH or filesystem operations." The boundary leaks both ways.

The status format has five independent implementations, two of them in the wrong file:

| Location | Form |
|---|---|
| `_jobs.py:280` | `JobStatus.to_json()` via `json.dumps(sort_keys=True)` |
| `_jobs.py:337-339` | shell `printf` hand-formatted to match that exact key order |
| `_jobs.py:467, 470, 477` | shell `grep -q '"state":"prepared"'` |
| `sandbox.py:663` | shell `grep -Eq '"state":"succeeded"\|"state":"failed"\|"state":"terminated"'` |
| `sandbox.py:738` | shell `sed -n 's/.*"pid":\([0-9][0-9]*\).*/\1/p'` |

Add a field to `JobStatus`, or drop `sort_keys`, and those greps break **silently** — a running job reads as terminal or vice versa, and no test catches it.

The fix is already demonstrated at `sandbox.py:667`, which correctly builds the terminated status as `JobStatus(...).to_json()` in Python and ships it as a quoted literal. Do that everywhere:
- Template `write_status` from `JobStatus(state, pid=..., returncode=...).to_json()` with `%s` placeholders substituted, rather than hand-writing the JSON object.
- Derive the terminal grep pattern: `"|".join(f'"state":"{s.value}"' for s in JobState if s.terminal)`.

Also move the `_signal_job` / `_cleanup_job` shell builders (`sandbox.py:616-776`, ~160 lines) into `_jobs.py` next to `launcher_script` and `submission_command`. They currently hardcode the protocol from outside it: `"$job/status.json"`, `"$job/termination.request"`, `.status.terminate.$$`, and the container-side `<dir>/pid`.

### 5. Delete the `execution.claim` protocol; use the `flock` that is already required

`submission_command` already hard-requires `flock` (`_jobs.py:451`) and uses it for `submission.lock`. Given that, the hand-rolled claim machinery is redundant:

- `_jobs.py:358-364` — write `.execution.claim.$$`, `ln` it, `rm` the candidate
- `_jobs.py:480-484` — `cat` the claim, parse the PID, `kill -0` for liveness
- `_jobs.py:485-491` — the `launch_round` loop that `rm -rf`s a stale claim and re-launches
- `_jobs.py:124-126` — the `execution_claim` path property

Replace with: the launcher takes `flock -n` on its own lock file for its lifetime and exits 0 if it cannot. "Is a launcher alive?" becomes a kernel-maintained fact **released automatically on death** — exactly what the `kill -0` probe and stale-claim recovery reconstruct by hand, with a PID-reuse race the kernel does not have.

Use a lock file distinct from `submission.lock` (e.g. `run.lock`) so the launcher does not contend with its submitter. ~35 lines of the trickiest shell in the branch go away and the result is more robust, not less.

### 6. `RemoteJobPaths` must own every path it claims to own

Its docstring says "All remote paths owned by one job" (`_jobs.py:97`), but these live as string literals in other modules: `submission.lock`, `.status.$$`, `.execution.claim.$$`, `.status.terminate.$$`, and the container job directory (`_container_job_directory`, `sandbox.py:1794`). Move them in, or fix the docstring — currently it is false.

### 7. Split `test/thunder_test.py` (1827 -> 3256 lines)

`AsyncSandboxTest` alone spans lines 1457-2769. The three new classes are self-contained and cover lines 120-800:

- `DurableJobProtocolTest` (`:120`) -> `test/durable_jobs_test.py`
- `SSHConnectionManagerTest` (`:388`) -> `test/ssh_manager_test.py`
- `DurableProcessTest` (`:487`) -> `test/durable_process_test.py`

CI discovers `*_test.py` (`ci.yml:50`) so new files run automatically. `BUILD.bazel` enumerates `srcs` individually — add the new files there too. Mechanical move; shared fixtures go in `test/ssh_faults.py` or a sibling helper.

### 8. `BUILD.bazel` references a file that does not exist

`BUILD.bazel:18` adds `"CHANGELOG.md"` to the `package_metadata` filegroup. There is no `CHANGELOG.md` in the repo and it is not gitignored, so `bazel build //:package_metadata` fails. Add the file or drop the line.

---

## Should fix

### 9. Move `OutputMode` to `_common/types.py`

`synchronous/sandbox.py:10` does `from ..asynchronous._jobs import OutputMode` and puts it in eleven public `exec` signatures. The synchronous package depends on a *private* module of the asynchronous package for part of its own *public* contract. `OutputMode` belongs in `_common/types.py` next to `GPUType` and `NetworkPolicy` — the module that exists for exactly this.

### 10. Make `spec` required in `launcher_script`

`_jobs.py:312` — `launcher_script(command, paths, spec=None)`. Every production call passes a spec (`_jobs.py:426`); only tests pass `None`. The `None` case forces `spec is None or spec.stdout == "capture"` twice at `_jobs.py:324-325`. Make it required and have the eight test call sites construct a two-line `JobSpec`.

### 11. `JobSpec` persists three fields nothing reads back

`_read_job_spec` -> `_durable_process` (`sandbox.py:488-526`) consumes only `job_id`, `container`, `stdout`, `stderr`, `retain`. `argv`, `workdir`, and `env` are written to `spec.json` and never read — the real command lives independently in `launch.sh`. That makes ~45 lines of the `from_json` type-checking ladder (`_jobs.py:227-247`) validation with no consumer.

Either drop those fields, or keep them and say in the docstring that they exist for operator debugging. As written, the file reads as though recovery depends on them.

### 12. Extract a shared base for the two readers

`_PreservingReader` and `_DurableReader` in `process.py` duplicate the buffering logic. `process.py:146-148` and `process.py:260-262` are byte-identical line-splitting; the `_empty` / `_newline` / `cast(T, ...)` init is duplicated; `read`, `readline`, `__aiter__`, `__anext__` share a shape.

Extract `_BufferedReader[T]` owning `_buffer`, `_empty`, `_newline`, `read`, `readline`, `__anext__` in terms of an abstract `_fill()` plus `_eof`. The pump-based and fetch-based readers then differ only in `_fill`. ~60 lines and one copy of the trickiest slicing logic go away.

### 13. Stop building `_signal_job` shell by splicing conditional fragments

`sandbox.py:668-745` — `container_signal`, `host_signal`, `workload_alive`, and `kill_completion` are string variables interpolated into a template, each with a host and a container variant. Conditionals that produce code are the hardest thing to review or test.

The host/container difference is one concept — "how do I address this job's process group" — and wants to be a small strategy (two functions returning a `kill` / `kill -0` prefix), not four spliced strings.

### 14. One parser for the `/.` contents-only convention

Re-parsed four different ways, with four separate re-derivations of the base name:

- `sandbox.py:791, 799` — `source.endswith(os.sep + ".")`
- `sandbox.py:846` — `remote_path.endswith("/.")`
- `sandbox.py:1055` — `raw_source.endswith(os.sep + ".")`
- `sandbox.py:1119` — `remote_path.endswith("/.")`

Missing model. Add one `_parse_transfer_path()` returning `(path, contents_only, name)`.

### 15. De-duplicate the contents-upload path

`_upload_guest_contents` (`sandbox.py:953`) and the `contents_only` branch of `_upload_to_container` (`sandbox.py:1055-1076`) are near-verbatim copies: reset staging dir, iterate `source_directory.iterdir()`, `scp` each child. Additionally `_upload_to_container` defines two nested closures in its if/else that differ only in which `scp` call they make — one closure with a parameter does both.

### 16. Convert shell string-matching tests to execution tests

`test_submission_is_detached_and_reuses_one_execution_claim` (`thunder_test.py:206-234`) makes twelve `assertIn` checks against exact generated shell fragments — `'ln -- "$claim_candidate" ...'`, `'flock -w 15 9'`, `"trap ':' HUP INT TERM"`. None pin a behavior; they pin an implementation and will fail on changes that alter nothing observable. **They will actively block items 4 and 5.**

`test_launcher_executes_a_payload_at_most_once` (`thunder_test.py:236-261`) is the right shape and already in the file: it runs the script twice and asserts one side effect. Follow that pattern.

---

## Minor

- **`_connect()` identity closure** (`sandbox.py:1192`): passes `async def connected(c): return c` to `run()` purely to borrow the retry loop. It works but reads like a mistake. Give `SSHConnectionManager` an explicit `connect(name)` method.
- **`__del__` schedules an unbounded task** (`process.py:545`): `self.cleanup()` calls `_wait_durable()`, which polls without a deadline. A finalizer resurrecting the object into a possibly-forever task is fragile. Bound it, or drop the finalizer in favour of the explicit `cleanup()` the README already documents.
- **`_consume_cleanup_result` swallows everything** (`process.py:549-553`): silent by design, but a `warnings.warn` or debug log would make a persistently-failing cleanup discoverable rather than invisible.
- **Stale attribution tags** (pre-existing, not from this branch): `(by claude)` at `sandbox.py:1383`, `sandbox.py:1400`, and near `sandbox.py:1683`. Strip them when touching those docstrings. Never add new ones.
