# Contributing to markovtrace

## Development setup

```bash
uv sync --extra dev --extra test
pre-commit install
```

The `uv sync --extra dev --extra test` command resolves and installs the project plus the `dev` and `test` optional dependency groups, and writes or updates `uv.lock`. The lockfile is committed; see ADR-0001 in the CHAINS coordination repository.

## Run the gates locally

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/
uv run bandit -c pyproject.toml -r src/
uv run pytest --cov=src/markovtrace --cov-fail-under=80
```

The `Makefile` wraps these: `make lint`, `make test-cov`, `make ci-local`.

## TDD discipline

Test commits precede feat commits for the same task ID. Adapter and eval tasks (E3, E6) use attack-first TDD: negative tests (auth failures, malformed responses, rate-limit handling, edge cases) commit separately before the positive feature tests.

Reference: CHAINS `CLAUDE.md` Rules 9 and 10 in the coordination repository.

## Commit format

Conventional commits with a task-ID prefix:

```
feat(E3): implement Anthropic adapter
test(E3): add negative tests for Anthropic adapter auth failures
docs(E3): document adapter interface
```

Every commit ends with a `Co-Authored-By` trailer:

```
Co-Authored-By: Claude <Model> (<context window>) <noreply@anthropic.com>
```

Atomic commits. The body explains why, not what.

## PR process

One PR per task. Reviewer agents (qa, supply-chain) run automatically against the PR. PM merge happens via `make merge-pr PR=<N>` in the CHAINS coordination repository; markovtrace inherits the merge convention. Bare `gh pr merge` is forbidden.

## Code of conduct

TBD. A standard CoC lands at v0.1.0.
