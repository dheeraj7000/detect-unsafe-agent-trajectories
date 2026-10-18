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
├── harness/            # Agent Control & Evaluation Lab (MCP gateway, sandbox, telemetry)
├── dataset/            # Labeled benign + adversarial long-horizon episodes
├── monitors/           # Rules, sequence, graph, and hybrid monitors
├── evaluator/          # Early-warning evaluation protocol & metrics
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

## Phases (24 weeks)

1. **Harness** (wks 1–4): MCP/tool gateway, sandbox, event schema, replay
2. **Dataset** (wks 5–8): Benign + adversarial long-horizon episodes with ground-truth labels
3. **Baselines** (wks 9–12): Rules + feature + sequence monitors
4. **Graph monitor** (wks 13–16): Online graph builder + GNN/graph-transformer
5. **Study** (wks 17–22): Matched-FPR early-detection experiments + ablations
6. **Release** (wks 23–24): Code, schemas, tasks, traces, paper/report

## Technical Stack

- Python agent harness; MCP-compatible proxy/gateway; Dockerized tool sandbox
- OpenTelemetry-style event schema + JSONL/Parquet trace store
- NetworkX (graph construction); PyTorch Geometric or DGL (graph models)
- Compact transformer / LLM embeddings; scikit-learn (simple baselines)
- Fixed seeds, immutable scenario manifests, deterministic replay

## See Also

- `docs/GraphGuard_Research_Plan.md` — full research & project plan
- `docs/` — behavior graph schema, evaluation protocol, threat scenarios
