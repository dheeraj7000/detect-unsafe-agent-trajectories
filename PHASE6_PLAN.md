# GraphGuard Phase 6 — Release Plan

**Status:** In progress. The core package, CLI, CI workflow, trace viewer, and
paper outline exist. Release validation and research-result documentation are
not complete.

## Deliverables

| # | Artifact | Current status | Remaining work |
|---|----------|----------------|----------------|
| 1 | Packaging | `pyproject.toml`, requirements files, and MIT license are present | Verify installation from a clean environment |
| 2 | CLI | `graphguard` console-script entry point is implemented and available in the current environment | Verify after clean install; `python -m graphguard` is not supported |
| 3 | CI | `.github/workflows/ci.yml` defines test, demo, CLI, lint, and Python 3.10–3.12 jobs | Confirm current workflow results |
| 4 | Trace viewer | `docs/_static/trace_viewer.html` is present | Verify it against a representative trace in a browser |
| 5 | Paper/report | `docs/PAPER_OUTLINE.md` is present as an outline | Run and record experiments; replace placeholder result claims with supported findings |
| 6 | Release checklist | `docs/RELEASE_CHECKLIST.md` is present | Complete remaining validation and release tasks |

## Current validation

- `python -m pytest tests/ -q`: **34 passed** in the available Python 3.13 environment.
- `graphguard --help` and `graphguard run --help`: pass in the current environment.
- `black --check` and `ruff check`: currently report issues.
- `python -m graphguard`: not supported; use the installed `graphguard` command.
- A clean-environment install, the CI matrix, demos, and trace viewer have not
  been verified as part of this status update.

## Remaining work

1. Run and record the planned H1–H5 experiments, including matched-FPR
   comparisons, utility, horizon/generalization analyses, ablations, and
   statistical results.
2. Update the paper outline to report only findings supported by those results.
3. Resolve formatting and lint findings; verify clean installation, demos,
   trace viewer, and CI on the supported Python versions.
4. Prepare release artifacts and create the v0.1.0 tag and GitHub release.
