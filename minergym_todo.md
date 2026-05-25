<!-- -*- mode: markdown -*- -->

# Subprocess isolation for EnergyPlus — implementation spec

**Context.** Even after the B0 resource-leak fix (upstream
`EnergyPlusEnvironment.close()` stops the thread, calls `delete_state`,
and rmtrees the output dir), EnergyPlus
accumulates ~14 MB/cycle of RSS in C++ global/static objects inside the DLL that
`delete_state` and `reset_state` cannot reach.  See `notes.md` §
"Residual EnergyPlus-native RSS growth".  The only complete fix is process
isolation: each simulation runs in a subprocess, and the OS reclaims all
EnergyPlus-internal memory when the process exits.

This document describes the changes needed in two parts:

- **Part 1** — new code in the `minergym` fork
- **Part 2** — integration in `building2building`

The public Gymnasium interface (`reset`, `step`, `close`) must not change.
Existing callers — `train_sac.py`, `tune_controller.py`,
`ResampleBuildingOnResetWrapper`, etc. — must require zero modification.

---

## Part 1 — minergym fork

### 1.1  New file `minergym/subprocess_worker.py`

This module runs **inside the worker subprocess**.  It has no imports from
`building2building`; it only depends on `minergym` itself.

#### Function `run_eplus_worker`

```python
def run_eplus_worker(
    make_energyplus: Callable[[], EnergyPlusSimulation],
    conn: multiprocessing.connection.Connection,
) -> None:
```

**Responsibility.** Create a simulation, start it, and relay observations and
actions between the EnergyPlus thread (inside this process) and the main process
(via `conn`).

**Protocol** (all messages are `(tag, payload)` tuples sent with `conn.send` /
`conn.recv`, using pickle):

| Direction | Message | Meaning |
|-----------|---------|---------|
| worker → main | `("obs", obs, False)` | Observation at current timestep |
| worker → main | `("obs", obs, True)` | Final observation, episode done |
| worker → main | `("error", exception)` | Simulation crashed |
| main → worker | `("action", action)` | Apply this action and advance |
| main → worker | `("stop",)` | Shut down the simulation and exit |

**Implementation sketch:**

```python
def run_eplus_worker(make_energyplus, conn):
    try:
        sim = make_energyplus()
        obs, done = sim.start()
        conn.send(("obs", obs, done))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                sim.try_stop()
                return
            if msg[0] == "action":
                obs, done = sim.step(msg[1])
                conn.send(("obs", obs, done))
                if done:
                    # Episode ended naturally; wait for stop or next action.
                    # try_stop() on the sim is not needed — state is StateDone.
                    pass
    except Exception as exc:
        try:
            conn.send(("error", exc))
        except Exception:
            pass
```

**Notes.**
- `make_energyplus` must be picklable (checked in § Part 2).
- The worker never imports `building2building`.  It only calls the factory and
  drives `EnergyPlusSimulation.step()`.
- EnergyPlus is loaded from scratch in the subprocess (spawn context) so there
  is no shared DLL state with the parent.

---

### 1.2  New file `minergym/subprocess_simulation.py`

#### Class `SubprocessEnergyPlusSimulation`

```python
@dataclass
class SubprocessEnergyPlusSimulation:
    make_energyplus: Callable[[], EnergyPlusSimulation]
    start_method: str = "spawn"   # "fork" is faster but unsafe with loaded DLL
    worker_timeout: float = 15.0
```

This class exposes the **same public interface** as `EnergyPlusSimulation`
(`.start()`, `.step()`, `.try_stop()`, `.stop()`) so it can be used as a
drop-in wherever `EnergyPlusSimulation` is expected.

**Instance state** (set by `start()`, cleared by `try_stop()`):

```python
_proc: multiprocessing.Process | None
_conn: multiprocessing.connection.Connection | None
_done: bool   # True after episode ended naturally
```

**`start(self) -> tuple[obs, bool]`**

1. Create a `multiprocessing.Pipe()` → `(parent_conn, child_conn)`.
2. Start a `Process(target=run_eplus_worker, args=(self.make_energyplus, child_conn),
   daemon=True)` using the configured start method.
3. Close `child_conn` in the parent (important — prevents the pipe from staying
   open indefinitely if the child crashes without sending).
4. `parent_conn.recv()` — blocks until the worker sends the first observation
   (or an error).
5. If `("error", exc)` received, raise `exc`.
6. Store `_proc`, `_conn = parent_conn`, `_done = done`.
7. Return `(obs, done)`.

**`step(self, action) -> tuple[obs, bool]`**

1. If `_done`, return `(last_obs, True)` immediately (matches `EnergyPlusSimulation`
   behaviour for `StateDone`).
2. `_conn.send(("action", action))`.
3. `_conn.recv()` — `("obs", obs, done)` or `("error", exc)`.
4. If error, call `_terminate_worker()`, raise.
5. Update `_done`.  Return `(obs, done)`.

**`try_stop(self)`**

1. If no worker is running, return immediately.
2. Try `_conn.send(("stop",))` (ignore BrokenPipeError if worker already died).
3. `_proc.join(timeout=self.worker_timeout)`.
4. If still alive, `_proc.kill(); _proc.join(timeout=2)`.
5. `_conn.close()`.
6. Set `_proc = None`, `_conn = None`.

**`stop(self)`** — alias for `try_stop()` (keeps interface parity with
`EnergyPlusSimulation.stop()`).

**`_terminate_worker(self)`** — internal; unconditionally kills the worker and
closes the connection.  Used in error paths.

---

### 1.3  Picklability requirement on `make_energyplus`

`run_eplus_worker` is passed to a subprocess via pickle.  The factory callable
must be picklable.  `EnergyPlusSimulation` is a `@dataclass(slots=True)` with
`Path`, `int`, `float`, and optree PyTree fields.  Verify with:

```python
import pickle
pickle.dumps(make_energyplus_instance)   # must not raise
```

`optree` PyTrees containing `VariableHole`, `MeterHole`, `ActuatorHole`
(frozen dataclasses) should pickle fine.  The reward function and observation
transform stored on `EnergyPlusEnvironment` are **not** passed to the worker —
the worker only runs `EnergyPlusSimulation`, not the full `EnergyPlusEnvironment`.

---

### 1.4  Start method: `spawn` vs `fork`

Use **`spawn`** by default.

`fork` copies the entire process including the loaded EnergyPlus DLL, any
Python threads, and any partially-initialised C++ global objects.  EnergyPlus
has global state (output variable registries, etc.) that may be in an
inconsistent state at fork time.  `spawn` starts a clean Python interpreter
and imports minergym fresh, guaranteeing isolation.

Cost: `spawn` takes ~0.5–2 s on a SLURM CPU node (one-time per episode, not
per step).  A winter episode is ~2160 steps × ~0.02 s = ~43 s, so the startup
overhead is <5%.

If `fork` is needed for performance (e.g., vec_env with very short episodes),
expose `start_method="fork"` as a parameter and document the risk.

---

## Part 2 — `building2building`

### 2.1  `building2building/simulator/__init__.py`

#### New boolean flag `use_subprocess` on `MakeEnergyPlus`

```python
@dataclass
class MakeEnergyPlus:
    ...
    use_subprocess: bool = False      # add this field
    subprocess_start_method: str = "spawn"
```

#### Updated `MakeEnergyPlus.__call__`

```python
def __call__(self) -> EnergyPlusSimulation | SubprocessEnergyPlusSimulation:
    sim = EnergyPlusSimulation(
        self.path_to_building,
        self.path_to_weather,
        self.observation_template,
        self.action_template,
        verbose=self.verbose,
        log_dir=self.log_dir,
        warmup_phases=self.warmup_phases,
        max_steps=self.max_steps,
    )
    if self.use_subprocess:
        from minergym.subprocess_simulation import SubprocessEnergyPlusSimulation
        return SubprocessEnergyPlusSimulation(
            make_energyplus=lambda: sim,
            start_method=self.subprocess_start_method,
        )
    return sim
```

Wait — this won't work cleanly because the lambda captures `sim`, and `sim` is
an `EnergyPlusSimulation` that needs to be created **inside** the subprocess
(so that EnergyPlus's native library initialises from scratch there).  The
factory passed to `SubprocessEnergyPlusSimulation` must itself create the
simulation when called inside the subprocess, not wrap an already-created one.

**Correct approach:** pass `self` (the `MakeEnergyPlus` dataclass) as the
factory, with `use_subprocess=False` so the worker doesn't recurse:

```python
def __call__(self) -> EnergyPlusSimulation | SubprocessEnergyPlusSimulation:
    if self.use_subprocess:
        from minergym.subprocess_simulation import SubprocessEnergyPlusSimulation
        factory = dataclasses.replace(self, use_subprocess=False)
        return SubprocessEnergyPlusSimulation(
            make_energyplus=factory,
            start_method=self.subprocess_start_method,
        )
    return EnergyPlusSimulation(
        self.path_to_building,
        self.path_to_weather,
        self.observation_template,
        self.action_template,
        verbose=self.verbose,
        log_dir=self.log_dir,
        warmup_phases=self.warmup_phases,
        max_steps=self.max_steps,
    )
```

`dataclasses.replace(self, use_subprocess=False)` creates a copy of
`MakeEnergyPlus` with `use_subprocess=False`, which is the factory that runs
inside the subprocess and returns a plain `EnergyPlusSimulation`.  This is
fully picklable.

#### Updated `create_simulator` signature

Add a `use_subprocess: bool = False` parameter and thread it through to
`MakeEnergyPlus`.

#### Upstream `EnergyPlusEnvironment.close()` interaction

When `use_subprocess=True`, `self.ep` is an `EnergyPlusSimulation` **from the
main process's perspective** but backed by a subprocess.  The upstream
`close()` calls `self.ep.try_stop()`, which — for
`SubprocessEnergyPlusSimulation` — sends `("stop",)` and kills the worker
process.  **No other changes needed** in `close()`: thread joining, `gc.collect()`,
and rmtree still apply (the thread join becomes a process join, handled inside
`try_stop()`; the output dir is still cleaned up by rmtree).

One addition: `close()` should **not** wait for the EnergyPlus thread to join
(there is none in the parent process).  The existing code already does this
safely because `ep_thread = getattr(ep_sim_state, "ep_thread", None)` will
return `None` for `SubprocessEnergyPlusSimulation` (it has no `.state` in the
minergym sense), so the thread-join branch is skipped.

---

### 2.2  `building2building/api/__init__.py` — `new_make_env`

Add a `use_subprocess: bool = False` parameter and pass it to
`create_simulator`.  Default `False` preserves all existing behaviour.

```python
def new_make_env(
    ...
    use_subprocess: bool = False,
) -> gym.Env:
```

---

### 2.3  `building2building/envs/factory.py` — `make_env_from_config`

Same: add `use_subprocess: bool = False` to `EnvBuildConfig` (or as a direct
parameter) and pass through to `create_simulator`.

---

### 2.4  Tests

#### `tests/long/test_env_leak.py`

Add a new test class `TestEnvLeakSubprocess` that mirrors `TestEnvLeakClose`
but creates envs with `use_subprocess=True`:

```python
_SUBPROCESS_ENV_KWARGS = {**_ENV_KWARGS, "use_subprocess": True}

class TestEnvLeakSubprocess:
    def test_subprocess_close_removes_output_dir(self): ...
    def test_subprocess_close_bounds_rss_growth(self): ...
```

The RSS test should use a much tighter limit (e.g. 30 MB total for N=20) since
subprocess isolation eliminates the ~14 MB/cycle growth.  The filesystem and
thread tests should pass unchanged.

#### `tests/quick/` — picklability smoke test

A quick test (no EnergyPlus needed, mock the simulation) that calls
`pickle.dumps(make_energyplus_instance)` and asserts it does not raise.

---

### 2.5  `notes.md` update

Once the subprocess mode passes the RSS test:
- Update the "Residual EnergyPlus-native RSS growth" gotcha to note that
  `use_subprocess=True` eliminates it.
- Update the SLURM memory budget guidance.

---

## Summary of files changed

| File | Change |
|------|--------|
| `minergym/subprocess_worker.py` | **New** — worker function |
| `minergym/subprocess_simulation.py` | **New** — `SubprocessEnergyPlusSimulation` |
| `building2building/simulator/__init__.py` | Add `use_subprocess` to `MakeEnergyPlus`; update `create_simulator` |
| `building2building/api/__init__.py` | Add `use_subprocess` param to `new_make_env` |
| `building2building/envs/factory.py` | Add `use_subprocess` param to `make_env_from_config` |
| `tests/long/test_env_leak.py` | Add subprocess test class |
| `tests/quick/` | Add picklability smoke test |
| `notes.md` | Update RSS growth entry once passing |

## Non-goals

- Changing the public `gym.Env` interface or any training script.
- Vectorised subprocess envs (`SubprocVecEnv`): `stable-baselines3` already
  provides this; the subprocess mode here is orthogonal (per-env isolation, not
  per-step parallelism).
- Supporting Windows (fork/spawn semantics differ; out of scope).
