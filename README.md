# GraphGuard

**Early Warning of Unsafe Long-Horizon Agent Behavior from Partial Execution Graphs**

Can graph-based representations detect unsafe long-horizon agent behavior earlier than ordinary trajectory classifiers?

## Research Objective

Build an instrumented agent-control testbed and test whether **partial execution graphs** provide earlier, better-calibrated warning of unsafe behavior than sequence-based trajectory classifiers, while preserving useful task performance.

## Core Hypothesis

Long-horizon agent failures are often **relational rather than local**: a tool call may be harmless alone but risky when connected to an untrusted input, sensitive resource, privilege change, memory update, or later external action. A typed temporal graph exposes these dependencies explicitly, so a graph monitor may reach a reliable warning threshold with fewer observed actions than a flat sequence classifier.

## Primary Research Question

At matched false-positive rate and task utility, does a graph-based monitor detect an unsafe trajectory at a smaller fraction of the episode than strong sequence baselines?

## Repository Structure

```
graphguard-work/
├── harness/            # Agent Control & Evaluation Lab (Phase 1)
├── dataset/            # Labeled benign + adversarial episodes (Phase 2)
├── monitors/           # Rules, sequence, temporal, and graph monitors (Phases 3–4)
├── evaluator/          # Early-warning evaluation protocol & metrics (Phase 5)
├── configs/            # Scenario manifests, hyperparameters, alert thresholds
├── docs/               # Research plan, schema, reports
├── notebooks/          # Exploration & analysis notebooks
└── tests/              # Unit & integration tests
```

## Hypotheses

| ID | Claim |
|----|-------|
| H1 | **Earlier warning**: graph monitors detect policy violations earlier than sequence monitors at matched FPR. |
| H2 | **Horizon robustness**: graph performance degrades less as irrelevant steps / episode length increase. |
| H3 | **Structural generalization**: graphs transfer better to unseen attack wording when the dependency pattern is preserved. |
| H4 | **Interpretability**: risk localizes to subgraphs/paths matching provenance or authorization violations. |
| H5 | **Hybrid value**: graph + sequence + deterministic policy checks outperform any single representation. |

## Decision Rule

Support the hypothesis only if the graph monitor produces **statistically meaningful earlier detection at matched FPR**, without materially reducing benign task utility, and the advantage persists on unseen scenarios or longer horizons. If accuracy improves but lead time does not, the central hypothesis is **not** supported.

## Progress

| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 1 — Harness | 1–4 | Gateway, sandbox, event schema, replay | Implementation complete |
| 2 — Dataset | 5–8 | Benign + adversarial episodes and labeling | Implementation complete |
| 3 — Baselines | 9–12 | Rules, feature, sequence, temporal monitors | Implementation complete |
| 4 — Graph monitor | 13–16 | Online graph builder and graph monitor | Implementation complete |
| 5 — Study | 17–22 | Evaluation, ablations, and statistics tooling | Implementation complete; research conclusions not yet established |
| 6 — Release | 23–24 | Release validation and publication artifacts | In progress |

The current test suite passes (50 tests in the available Python 3.13 environment). Ruff still reports style/import findings in the codebase. A two-seed Phase 5 smoke run did not support H1; it has only three held-out attack episodes per seed, so it is not evidence for a research conclusion. The study code and paper outline are present, but empirical claims still need a sufficiently powered, preregistered run.

## Phase 1 — Harness (complete)

**Exit criterion met:** deterministic replay of 3 task families (`file_backup`, `email_triage`, `code_review`).

Modules: `schema.py`, `telemetry.py`, `gateway.py`, `sandbox.py`, `agent.py`, `graph.py`, `orchestrator.py`, `replay.py`, `scenarios.py`

- MCP-compatible tool proxy with identity + policy engine (6 rules: credential write, external send, dangerous code, memory poison, admin write, secret read)
- Mock sandboxes: file, email, code
- Online typed behavior-graph builder (NetworkX)
- Deterministic replay verified across 3 runs per family

## Phase 2 — Dataset (complete)

Labeled benign + adversarial episodes across **6 threat scenarios**, all with ground-truth labels derived from environment state + explicit policy (no LLM judge):

| Scenario | Attack family | Severity |
|----------|--------------|----------|
| Indirect prompt injection / tool-output hijacking | `indirect_prompt_injection` | high |
| Memory poisoning | `memory_poisoning` | high |
| Privilege escalation / permission probing | `privilege_escalation` | critical |
| Sensitive-data exfiltration | `data_exfiltration` | critical |
| Goal drift / specification gaming | `goal_drift` | medium |
| Multi-agent delegation | `multi_agent_delegation` | high |

Modules: `generator.py`, `labeler.py`, `store.py`, `synthetic.py`, `LABELING_RUBRIC.md`

- `generator.py`: episode generator; each adversarial episode is flagged by a policy denial at a known onset step
- `labeler.py`: ground-truth labels (task_success, policy_violation, unsafe_event_step, attack_family, affected_asset, trust_boundary, severity) + graph detection signals
- `store.py`: JSONL episode store + index + Parquet export
- `synthetic.py`: provenance / authorization / destination variant pairs for H3 structural generalization

## Phase 3 — Baselines (complete)

Ordinary-trajectory baselines, all sharing the `BaseMonitor` interface
(`score_prefix` → risk score in [0,1]; `calibrate` → FPR thresholds at 1%/5%;
early-detection metrics in `calibrator.py`).

Modules: `base.py`, `rules.py`, `feature.py`, `sequence.py`, `temporal.py`,
`transformer.py`, `calibrator.py`, `registry.py`

| Monitor | Input | Role |
|---------|-------|------|
| rules | gateway policy decisions | High-precision interpretable baseline |
| feature | aggregated trace statistics | Cheap classical baseline (LogisticRegression) |
| sequence | serialized trajectory prefix | Primary ordinary-trajectory baseline (TF-IDF + LR) |
| temporal | event embeddings over time | Strong long-context baseline (sklearn MLP, optional torch) |
| transformer | serialized event tokens in temporal order | Trainable Transformer encoder sequence baseline |

Each monitor is calibrated on benign validation traces so that alert
thresholds hit target false-positive rates (1% and 5%). Early-detection
metrics (lead time, normalized detection time, TPR@FPR) are demonstrated in
`demo_phase3.py`.

## Phase 4 — Graph Monitor (complete)

The proposed GraphGuard method: a typed message-passing GNN that scores risk
on the partial behavior graph after every action. It processes graph snapshots
with node and edge step features; it does not maintain recurrent hidden state
across snapshots.

Modules: `graph_model.py`, `graph.py` (in `monitors/`), plus
`graph_to_tensors()` in `harness/graph.py`.

| Component | Purpose |
|-----------|---------|
| TemporalGraphNN | Pure-PyTorch message-passing GNN: typed node embeddings and typed/time-aware edge embeddings propagated in K message-passing layers, global mean pooling, sigmoid risk head |
| GraphMonitor | BaseMonitor-compatible wrapper; scores risk from the graph; fit() trains on labeled prefixes via binary cross-entropy |
| graph_to_tensors() | Converts BehaviorGraph → node/edge tensors (typed one-hot + step normalization) |

The graph monitor receives explicit provenance / authorization / dependency
edges and their step features, so it can model relational risk patterns (e.g. untrusted tool
output → plan update → credential read → external send) earlier than a
flat sequence model. `demo_phase4.py` demonstrates calibrated early
detection across the 6 threat scenarios and compares lead time against
the sequence baselines.

## Phase 5 — Study tooling (implemented)

Matched-FPR early-detection study comparing the graph monitor against
sequence baselines, with hypothesis testing (H1–H5), ablations, and
counterfactual replay.

Modules: `study.py`, `statistics.py`, `ablations.py`, `counterfactual.py`
(in `evaluator/`), plus `protocol.py` and strict split helpers in
`dataset/splits.py`.

| Component | Purpose |
|-----------|---------|
| StudyRunner | End-to-end study: fit → calibrate → evaluate → aggregate → statistical comparison |
| statistics.py | Bootstrap CIs, paired t-test, Wilcoxon signed-rank, CI for difference |
| ablations.py | Graph structure variants (no provenance, no authorization, no temporal, untyped, bounded window) |
| counterfactual.py | Counterfactual replay: remove dangerous dependency → risk should drop |

Key metrics: normalized detection time, lead actions, TPR@FPR, detect rate,
AUROC/AUPRC, monitoring overhead, calibration (ECE/Brier).

Hypothesis evaluation:
- **H1 (Earlier warning)**: graph lead time > sequence lead time at matched FPR
- **H2 (Horizon robustness)**: stratify by episode length
- **H3 (Structural generalization)**: train on one attack family, test on unseen
- **H4 (Interpretability)**: counterfactual replay — risk localizes to provenance/auth paths
- **H5 (Hybrid value)**: graph + sequence + rules Pareto frontier

`demo_phase5.py` demonstrates the full study: calibration at FPR=5%, paired
statistical comparisons (bootstrap CI + paired t-test + Wilcoxon), ablation
study, and counterfactual replay. It supports multiple seeds with
`python demo_phase5.py --seeds 42 43`; the protocol and train/validation/test
episode IDs are frozen before fitting and test scoring. Training, calibration,
and test templates and wording variants are kept disjoint. On the current
default training split, the graph and Transformer models have approximately
matched capacity (about 108k vs. 113k parameters).

Implementation of the study tooling does not by itself establish H1–H5. No
empirical conclusion should be treated as supported until experiment outputs
are recorded, reviewed, and reflected in the paper.

## How to run

```bash
# Phase 1 demo: 3 task families + deterministic replay
python demo.py

# Phase 3 demo: baseline monitors + early detection
python demo_phase3.py

# Phase 4 demo: graph monitor vs sequence baselines
python demo_phase4.py

# Phase 5 demo: matched-FPR early-detection study
python demo_phase5.py

# Run all tests
python -m pytest tests/ -v
```

## Installation

```bash
# Clone and install
git clone https://github.com/dheeraj7000/detect-unsafe-agent-trajectories.git
cd detect-unsafe-agent-trajectories
pip install -e .

# Verify CLI
graphguard --help
graphguard run --scenario file_backup --monitor graph --steps 10
graphguard compare --scenarios data_exfiltration --monitors graph,sequence --fpr 0.05
graphguard calibrate --monitor graph --fpr 0.01 0.05
graphguard ablate --scenarios data_exfiltration --fpr 0.05
```

## Trace viewer

Open `docs/_static/trace_viewer.html` in a browser and load a trace file from `data/traces/*.jsonl` to step through an episode and inspect the partial behavior graph at each step.


## Minimal Technical Stack

- Python agent harness; MCP-compatible proxy/gateway; Dockerized tool sandbox
- OpenTelemetry-style event schema + JSONL/Parquet trace store
- NetworkX (graph construction); PyTorch (GNN graph model; PyTorch Geometric optional)
- Compact transformer / LLM embeddings; scikit-learn (simple baselines)
- Fixed seeds, immutable scenario manifests, deterministic replay

## See Also

- `docs/GraphGuard_Research_Plan.md` — full research & project plan
- `dataset/LABELING_RUBRIC.md` — labeling principles and ground-truth sources
- `configs/default.yaml` — default harness configuration

## Repo

`https://github.com/dheeraj7000/detect-unsafe-agent-trajectories.git`
