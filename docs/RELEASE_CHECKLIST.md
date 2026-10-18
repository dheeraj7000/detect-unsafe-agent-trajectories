# GraphGuard Release Checklist

**Target:** v0.1.0 public release

## Packaging
- [ ] `pyproject.toml` — project metadata, dependencies, entry points
- [ ] `requirements.txt` — core dependencies (pinned)
- [ ] `requirements-dev.txt` — dev/test dependencies
- [ ] `LICENSE` — MIT
- [ ] `pip install .` works from clean checkout
- [ ] `graphguard` CLI command available after install
- [ ] `graphguard run --help` shows usage

## Code quality
- [ ] `python -m pytest tests/` passes (34 tests)
- [ ] `black --check` passes
- [ ] `ruff check` passes
- [ ] `mypy` type check (best-effort)
- [ ] All demos run: `demo.py`, `demo_phase3.py`, `demo_phase4.py`, `demo_phase5.py`

## CI
- [ ] `.github/workflows/ci.yml` — runs tests on push/PR
- [ ] CI tests on Python 3.10, 3.11, 3.12
- [ ] CI runs demos as smoke tests
- [ ] CI runs CLI smoke tests

## Documentation
- [ ] `README.md` — install + usage instructions (all demo commands)
- [ ] `docs/GraphGuard_Research_Plan.md` — full research plan (updated through Phase 5)
- [ ] `docs/PAPER_OUTLINE.md` — paper outline (abstract, method, experiments, results)
- [ ] `docs/RELEASE_CHECKLIST.md` — this file
- [ ] `PHASE6_PLAN.md` — Phase 6 plan
- [ ] `docs/_static/trace_viewer.html` — static trace/graph viewer

## Release artifacts
- [ ] Git tag `v0.1.0`
- [ ] GitHub Release with release notes
- [ ] Archived traces & trained model checkpoints (or download links)
- [ ] Research report / paper draft

## Demo commands (verified working)
```bash
# Phase 1: 3 task families + deterministic replay
python demo.py

# Phase 3: baseline monitors + early detection
python demo_phase3.py

# Phase 4: graph monitor vs sequence baselines
python demo_phase4.py

# Phase 5: matched-FPR early-detection study
python demo_phase5.py

# CLI
graphguard run --scenario file_backup --monitor graph --steps 10
graphguard compare --scenarios data_exfiltration --monitors graph,sequence --fpr 0.05
graphguard calibrate --monitor graph --fpr 0.01 0.05
graphguard ablate --scenarios data_exfiltration --fpr 0.05
```
