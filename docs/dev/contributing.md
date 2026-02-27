# Contributing

Thank you for considering a contribution to Building2Building! This guide
explains the workflow for reporting issues, submitting changes, and getting
your code merged.

## Reporting Issues

- Search existing issues before opening a new one.
- Include a minimal reproducible example when reporting bugs.
- For feature requests, describe the use case and expected behaviour.

## Branch Naming

Use the following prefixes:

| Prefix       | Purpose                        |
|------------- |--------------------------------|
| `feat/`      | New features                   |
| `fix/`       | Bug fixes                      |
| `refactor/`  | Code restructuring             |
| `docs/`      | Documentation changes          |
| `test/`      | Adding or updating tests       |
| `ci/`        | CI/CD configuration            |

Example: `feat/multi-zone-reward`

## Submitting a Pull Request

1. Fork the repository and create your branch from `main`.
2. Install the project in development mode: `pip install -e ".[dev]"`
3. Make your changes, ensuring all tests pass.
4. Push your branch and open a Pull Request against `main`.
5. Fill in the PR template—describe *what* changed and *why*.

## PR Review Process

- At least one maintainer must approve before merging.
- CI checks (linting, type-checking, tests) must pass.
- Address review comments by pushing additional commits—do not force-push
  during review.

## Code Review Checklist

Before requesting review, verify:

- [ ] All public functions have type hints and docstrings.
- [ ] New code is covered by tests.
- [ ] `black` and `ruff` report no issues.
- [ ] No hardcoded file paths—use `DataPaths` or configuration.
- [ ] No unrelated changes are included in the diff.
