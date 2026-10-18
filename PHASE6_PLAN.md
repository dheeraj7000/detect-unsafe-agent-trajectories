# GraphGuard Phase 6 — Release Plan (Weeks 23–24)

**Goal:** Public, reproducible release of GraphGuard as a benchmark artifact.

## Deliverables

| # | Artifact | Description | Status |
|---|----------|-------------|--------|
| 1 | Packaging | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, MIT LICENSE | To build |
| 2 | CLI entry point | `python -m graphguard` / `graphguard` command: run episodes, compare monitors, export results | To build |
| 3 | CI | GitHub Actions: run tests on push/PR, lint, test on Python 3.10+ | To build |
| 4 | Trace viewer | Static HTML/JS dashboard to load a trace JSON + behavior graph and step through an episode | To build |
| 5 | Paper/report | `docs/PAPER_OUTLINE.md` + full research report (abstract, method, results, related work) | To build |
| 6 | Release checklist | `docs/RELEASE_CHECKLIST.md` | To build |

## Success criteria

- `pip install .` works from a clean checkout
- `graphguard run --help` shows the CLI
- `python -m pytest tests/` passes in CI
- Trace viewer can load a `data/traces/*.jsonl` file and render the episode graph
- Paper outline covers: abstract, related work (METR, AgentDojo, AgentDyn, AgentArmor), method, experiments (H1–H5), results, discussion

## Order of work

1. Packaging + CLI (foundational — everything else builds on it)
2. CI (protects the codebase)
3. Trace viewer (demonstration artifact)
4. Paper outline + report (publication artifact)
5. Release checklist + tag v0.1.0
