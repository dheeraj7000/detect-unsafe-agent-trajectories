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
├── harness/            # Agent Control & Evaluation Lab (Phase 1 ✅)
├── dataset/            # Labeled benign + adversarial episodes (Phase 2 ✅)
├── monitors/           # Rules, sequence, graph, hybrid monitors (Phase 3)
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
| 1 — Harness | 1–4 | MCP/gateway, sandbox, event schema, replay | ✅ Complete |
| 2 — Dataset | 5–8 | Benign + adversarial long-horizon episodes | ✅ Complete |
| 3 — Baselines | 9–12 | Rules + feature + sequence monitors | ✅ Complete |
| 4 — Graph monitor | 13–16 | Online graph builder + GNN/graph-transformer | ✅ Complete |
| 5 — Study | 17–22 | Matched-FPR early-detection + ablations | ✅ Complete |
| 6 — Release | 23–24 | Code, schemas, traces, paper/report | Planned |

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
`calibrator.py`, `registry.py`

| Monitor | Input | Role |
|---------|-------|------|
| rules | gateway policy decisions | High-precision interpretable baseline |
| feature | aggregated trace statistics | Cheap classical baseline (LogisticRegression) |
| sequence | serialized trajectory prefix | Primary ordinary-trajectory baseline (TF-IDF + LR) |
| temporal | event embeddings over time | Strong long-context baseline (sklearn MLP, optional torch) |

Each monitor is calibrated on benign validation traces so that alert
thresholds hit target false-positive rates (1% and 5%). Early-detection
metrics (lead time, normalized detection time, TPR@FPR) are demonstrated in
`demo_phase3.py`.

## Phase 4 — Graph Monitor (complete)

The proposed GraphGuard method: a temporal GNN that scores risk on the
partial typed behavior graph after every action.

Modules: `graph_model.py`, `graph.py` (in `monitors/`), plus
`graph_to_tensors()` in `harness/graph.py`.

| Component | Purpose |
|-----------|---------|
| TemporalGraphNN | Pure-PyTorch message-passing GNN: typed node/edge embeddings, K rounds of typed message passing, global mean pooling, sigmoid risk head |
| GraphMonitor | BaseMonitor-compatible wrapper; scores risk from the graph; fit() trains on labeled prefixes via binary cross-entropy |
| graph_to_tensors() | Converts BehaviorGraph → node/edge tensors (typed one-hot + step normalization) |

The graph monitor sees explicit provenance / authorization / dependency
edges, so it can detect relational risk patterns (e.g. untrusted tool
output → plan update → credential read → external send) earlier than a
flat sequence model. `demo_phase4.py` demonstrates calibrated early
detection across the 6 threat scenarios and compares lead time against
the sequence baselines.

## Phase 5 — Study (complete)

Matched-FPR early-detection study comparing the graph monitor against
sequence baselines, with hypothesis testing (H1–H5), ablations, and
counterfactual replay.

Modules: `study.py`, `statistics.py`, `ablations.py`, `counterfactual.py`
(in `evaluator/`).

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
study, and counterfactual replay.

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
