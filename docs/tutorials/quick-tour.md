# Quick Tour

This tutorial walks you through the core B2B API: creating environments,
inspecting observation/action spaces, running an episode, and exploring the
morphology graph.

**Runnable script:** `tutorials/quick_tour.py`

---

## What the script covers

1. **List building types** — `b2b.list_building_types()` returns the 6
   available building types.

2. **Create an environment** — `b2b.new_make_env("OfficeSmall", split="test",
   index=0, task="task_const_e0")` returns a Gymnasium-compatible env.

3. **Inspect metadata** — `env.metadata` exposes `observation_names`,
   `action_names`, `hvac_equipment`, and the `morphology` graph.

4. **Run a random-action episode** — call `env.reset()`, then loop with
   `env.action_space.sample()` until `terminated or truncated`.

5. **Explore the morphology graph** — `morph = env.metadata["morphology"]`;
   `morph.split_observation(obs)` returns a per-node dict of local
   observations.

6. **Compare building types** — iterate over
   `["SingleFamilyHouse", "OfficeSmall", "OfficeMedium"]` and print
   `observation_space.shape[0]` / `action_space.shape[0]` / `len(morph.nodes)`.

---

## Minimal example

```python
import building2building as b2b

env = b2b.new_make_env("OfficeSmall", split="test", index=0, task="task_const_e0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

For the full step-by-step walkthrough, run:

```bash
python tutorials/quick_tour.py
```
