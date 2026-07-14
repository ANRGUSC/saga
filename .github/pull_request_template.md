<!--
Thanks for contributing to SAGA. Fill in the sections below and check the boxes
before requesting review. See CONTRIBUTING.md for the full review criteria.
-->

## Summary

<!-- What does this change do, and why? Link any related issue with "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New scheduler
- [ ] New feature (other)
- [ ] Refactor / maintenance
- [ ] Docs / CI only

## Checklist

- [ ] Branched off `main` and up to date with it (no conflicts)
- [ ] `uv run pytest tests/ --timeout=120` passes locally
- [ ] `uv run mypy src/saga --ignore-missing-imports` is clean
- [ ] `uv run ruff check src/saga` and `uv run ruff format --check src/saga` are clean
- [ ] New or changed behavior is covered by tests
- [ ] Public functions/classes are typed and documented in the existing docstring style
- [ ] No stray files, debug prints, unintended `pyproject.toml`/`uv.lock` changes, or version bump

## Notes for reviewers

<!-- Anything worth flagging: design tradeoffs, follow-ups, areas you want a close look at. -->
