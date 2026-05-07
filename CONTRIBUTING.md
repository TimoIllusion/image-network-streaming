# Contributing

Thanks for your interest. This file is the short version; full conventions, branch and PR rules, commit-tag policy, beads workflow, and architecture pointers live in [AGENTS.md](AGENTS.md).

## Quick start

```bash
uv sync --all-extras   # install dev + test extras
uv run pytest -q       # run the test suite
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Workflow

- Branch off `main`, open a PR, squash-merge back. `main` stays deployable.
- Branch names: `<type>/<short-kebab-description>` (`feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `test/`, `ci/`, `hotfix/`).
- Commit and PR subjects use the format `type: subject #tag`, where `#tag` is one of `#none`, `#patch`, `#minor`, `#major` (read by `.github/workflows/auto-tag.yml` to drive `setuptools_scm` versioning).
- Run `uv run ruff check .` and `uv run pytest -q` before pushing; both must pass.
- Issue tracking uses [beads](https://github.com/williampma/beads): `bd ready`, `bd show <id>`, `bd update <id> --claim`, `bd close <id>`.

See [AGENTS.md](AGENTS.md) for the canonical rules.

## Reporting security issues

See [SECURITY.md](SECURITY.md) — please do not file public issues for vulnerabilities.
