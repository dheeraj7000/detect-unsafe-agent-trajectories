# GraphGuard Release Checklist

**Target:** v0.1.0
**Status:** In progress; the repository contains the release scaffolding, but
validation and release artifacts remain.

## Implemented or present

- [x] Package metadata, declared dependencies, and `graphguard` console-script entry point in `pyproject.toml`
- [x] Core and development dependency files
- [x] MIT license
- [x] CLI commands: `run`, `compare`, `calibrate`, and `ablate`
- [x] GitHub Actions workflow for tests, demos, CLI smoke tests, lint, and Python 3.10–3.12
- [x] Static trace viewer at `docs/_static/trace_viewer.html`
- [x] Research plan, paper outline, and this checklist
- [x] Test suite: 34 passed in the available Python 3.13 environment
- [x] `graphguard --help` and `graphguard run --help` in the current environment

## Still to verify or complete

- [ ] Clean-environment package installation and CLI verification
- [ ] CI results on Python 3.10, 3.11, and 3.12
- [ ] Black formatting check and Ruff lint check (both currently report issues)
- [ ] Full demo suite and CLI smoke commands
- [ ] Trace viewer loading and rendering a representative trace
- [ ] Run and record the planned H1–H5 experiments; the paper outline's result claims are placeholders until verified
- [ ] Final research report based on recorded results
- [ ] Decide whether to include archived traces and trained checkpoints in the release
- [ ] Create Git tag `v0.1.0`
- [ ] Publish GitHub release notes and release

`python -m graphguard` is not currently supported; use the installed `graphguard`
console command.

## Commands to verify before release

```bash
python -m pytest tests/ -v
black --check harness/ dataset/ monitors/ evaluator/ graphguard/ tests/ demo*.py
ruff check harness/ dataset/ monitors/ evaluator/ graphguard/ tests/ demo*.py
python demo.py
python demo_phase3.py
python demo_phase4.py
python demo_phase5.py
graphguard run --scenario file_backup --monitor graph --steps 10
graphguard compare --scenarios data_exfiltration --monitors graph,sequence --fpr 0.05
graphguard calibrate --monitor graph --fpr 0.01 0.05
graphguard ablate --scenarios data_exfiltration --fpr 0.05
```
