# Contributing to SAGA

This document describes how changes get into `main`. It is written for both authors opening a pull request and maintainers reviewing one.

## Branching model

- `main` is the stable, released branch and the integration base. It is protected: changes land only through a reviewed pull request, never a direct push.
- Branch new work off `main`. Use `feature/<short-name>` for features and `fix/<short-name>` for bugfixes.
- Keep a branch focused on one thing. If a branch grows a second unrelated change, split it.
- Delete a branch once its pull request is merged.

## Requirements for merging into `main`

A pull request may be merged only when all of the following hold. The first four are enforced by CI and branch protection; the rest are the reviewer's responsibility.

1. The branch targets `main` through a pull request.
2. All tests pass on every supported Python version (3.12 and 3.13).
3. `mypy` reports no errors.
4. `ruff check` and `ruff format --check` are clean.
5. At least one maintainer has approved the pull request.
6. The branch is up to date with `main` with no merge conflicts.

Run the same checks locally before requesting review:

```bash
uv run pytest tests/ --timeout=120     # tests (matches CI; timeout avoids slow brute-force/SMT hangs)
uv run mypy src/saga --ignore-missing-imports
uv run ruff check src/saga
uv run ruff format --check src/saga    # drop --check to apply fixes
```

## Reviewer checklist

### Correctness and scope

- The change does what its description says, and nothing unrelated rides along.
- Edge cases are handled: empty graphs, single-node networks, disconnected components, ties, zero and infinite weights. SAGA fills missing edges with convention values (0 or infinity); confirm downstream math survives them.
- Float comparisons use `EPS = 1e-9` from `saga/__init__.py` rather than exact equality.

### Tests

- New or changed behavior is covered by a test. A change with no test needs a reason.
- A new scheduler is exercised by the parametrized suite in `tests/test_schedulers.py` (it is picked up automatically once registered, see below).
- The full suite passes locally with the timeout, not just the targeted test.

### Adding a scheduler

- One file in `src/saga/schedulers/`, one class subclassing `Scheduler`.
- Implements `schedule(self, network, task_graph)` and returns a valid `Schedule`.
- Registered in `src/saga/schedulers/__init__.py` so the parametrized tests find it.
- Avoid `isinstance` checks on the algorithm inputs (see issue #31).

### Types

- Public functions and methods carry type annotations.
- `mypy` is clean. Do not silence errors with blanket `# type: ignore`; if an ignore is unavoidable, make it specific and comment why.
- Core model types stay frozen Pydantic v2 models with a custom `__hash__`.

### Documentation and style

- New public functions, classes, and methods have docstrings in the existing Google style (a one-line summary, then `Args:` and `Returns:` sections), matching the surrounding code.
- Naming, structure, and idiom match the file the code lives in. New code should read like the code already in `main`.
- Comments explain intent where it is not obvious, and are kept in sync with the code.
- No hard-wrapped prose in docs or docstrings; let lines soft-wrap.

### Hygiene

- No debug prints, commented-out blocks, or stray scratch files.
- No large data files or generated artifacts committed. Experiment and example scripts live under `scripts/` and are not shipped in the package.
- `pyproject.toml` and `uv.lock` changes are intentional. Do not repin the `wfcommons` fork (`jaredraycoleman/WfCommons`, `windows-support` branch) back to upstream; it is pinned on purpose.
- No version bump in a feature or fix pull request. Releases are cut separately via `release.sh`.

## Merging

- Prefer squash-and-merge so `main` keeps one commit per change with a clean message. Use a merge commit only when preserving the individual commits matters.
- Resolve every review conversation before merging.
- Delete the source branch after the merge.
