# markovtrace

Counterfactual attribution for LLM outputs. Token, chunk, and section granularity.

[![CI](https://github.com/aedile/markovtrace/actions/workflows/ci.yml/badge.svg)](https://github.com/aedile/markovtrace/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)](https://github.com/aedile/markovtrace)
[![PyPI](https://img.shields.io/badge/pypi-pending-lightgrey)](https://pypi.org/project/markovtrace/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

The package is not yet on PyPI. Once published:

```bash
pip install markovtrace
```

or with uv:

```bash
uv add markovtrace
```

### From source

```bash
git clone https://github.com/aedile/markovtrace.git
cd markovtrace
uv sync --extra dev --extra test
```

## Quickstart

```python
# v0.0.1 ships scaffolding only. The public API lands in v0.1.0 with E3.
```

## Status

Pre-alpha. Scaffolding only. Public API and adapters land in subsequent releases.

## Tiered coverage

Coverage gates follow ADR-0004 in the coordinating CHAINS repository. The `src/markovtrace/eval/` package and the attribution math modules gate at 95% line coverage. Adapters and plumbing gate at 80%. CI enforces the split with two `pytest --cov` runs in series, each scoped to one path set with its own `--cov-fail-under` value.

At v0.0.1 the `eval/` module is a placeholder docstring. The second CI step is therefore permissive until E6 lands testable code under `src/markovtrace/eval/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, local gates, TDD discipline, commit format, and PR process.

## License

MIT. See [LICENSE](LICENSE).
