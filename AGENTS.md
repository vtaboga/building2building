<!-- -*- mode: markdown -*- -->

# Building2Building - Agent Guidelines

## Project Overview
This is a reinforcement learning framework for EnergyPlus building simulation.
- Main package: `building2building/`
- Uses Python 3.10+, PyTorch, Gymnasium, and EnergyPlus
- Follows scientific Python conventions with type hints

## Associated Publication

This codebase accompanies the paper:

> *Building2Building: A Large Scale Benchmark for Generalizable Real-World
> Reinforcement Learning* — Taboga, Veilleux, Jang, Rankawat, Bacon.
> **Accepted at RLJ / RLC.** Source: `paper/main.tex`, `paper/main.bib`.

The codebase MUST stay consistent with the paper: experimental setup,
algorithms, hyperparameters, environments, benchmarks, and reported numbers
should all be reproducible from this repository.

Because the paper has been accepted, only **light camera-ready modifications**
are allowed (typos, clarifications, reviewer-requested wording changes,
formatting fixes). Any non-trivial change to the paper or to code that affects
results in the paper requires my explicit approval before being made. When in
doubt, ask.

## Working Principles

These take precedence over everything else in this file.

- **Do not make assumptions.** If instructions are unclear, ask as many
  clarifying questions as required to remove all ambiguity before acting.
- **This is a research codebase. Fail loudly.** Do not introduce or maintain
  silent fallbacks (default values, try/except that swallows errors, "best
  effort" branches) as the code changes. If something is wrong, raise.
- **Be explicit to the reader.** Do not use complex engineering patterns when
  simple ones suffice, unless explicitly asked for modular or scalable
  solutions. Prefer straight-line code over abstractions.
- **Preserve existing comments.** Do not remove previous comments unless the
  underlying logic changes and the comment is no longer correct.
- **Prioritize reproducibility and clarity over clever design and efficiency.**
- **Align with the project plan.** When designing solutions, ensure they are
  consistent with the high-level project plan. Consult `design_doc.md` and
  `paper/main.tex` before proposing architectural or experimental changes.
- **One TODO, one commit.** If `TODO.md` exists at the project root, each TODO
  item should be committed separately.
- **Never run anything on a Slurm login node.** By default the agent
  is on a small interactive compute node (1 CPU, 4 GB RAM) suitable
  only for short, light commands (git, file edits, parsing W&B
  summaries, ≤ a few-minute Python scripts, lint, small unit tests).
  Anything heavier — training runs, full evals, anything that needs a
  GPU, more cores, more memory, or more than a few minutes of wall-
  clock — must be packaged as a Slurm script
  with appropriate `#SBATCH` resource directives. **Agents must
  never `sbatch` jobs themselves.** Write the script, commit it,
  and tell the user the exact `sbatch ...` (or
  `bash scripts/experiment/submit_all.sh`) command to run. The user
  is the only one who submits to Slurm.
- **Track experiments via git, not via script duplication.** Each
  experiment batch (parameter choice, hyperparameter set, step
  budget, architecture variant, sweep scope) should correspond to a
  git commit (or a small series of commits), so any past run can be
  reproduced by checking out that commit and re-running. Do **not**
  fork scripts or configs into many near-duplicate copies to encode
  variation; prefer (a) editing the canonical script/config and
  committing, or (b) one short, readable script per cell that defers
  shared logic to the canonical training entry point.


## User Rules
- Please reply in a concise style. Avoid unnecessary repetition or filler language.

## Code Style & Standards

### Code format
- Use the black code formatter.
- Don't write lines with only whitespace and don't keep trailing whitespace at
  the end of a line.

### Use types

We want as many bugs as possible to be caught by the type checker.

- Always use type hints for function parameters and returns.
- Use `from typing import` for complex types.
- Prefer `pathlib.Path` over string paths.
- Use `@dataclass` over ad-hoc dicts.
- When ingesting untyped data (for instance, to put it into a record), fail as
  soon as possible instead of assuming the data has the right type. Write
  `load_json` class method and `save_json` in the dataclass definition.
- When only a finite (and reasonably sized) set of literal values are allowed,
  use `Literal[...]` instead of `str`.

### Error Handling
- Don't catch exceptions that are unrecoverable.
- Always validate user inputs early in functions.

### Testing
- Use pytest with descriptive test function names starting with `test_`.
- Use `pytest.raises()` for exception testing.
- Put test data in `tests/data/` directory.
- Mock external dependencies (EnergyPlus, file downloads, etc.).

### Paths
- Use `DataPaths` class for consistent path resolution.
- Again, use `Path` and never `str` for path variables.

## What NOT to do
- Don't use string concatenation for paths — use `Path` methods.
- Don't ignore type hints — this is a typed codebase.
- Don't create giant functions — break complex logic into smaller functions.
- Don't hardcode file paths — use `DataPaths` or configuration.
- Don't ignore existing abstractions (use `StateCode`, not raw strings).
- Don't commit hardcoded personal informations (API keys, specific paths etc.)
