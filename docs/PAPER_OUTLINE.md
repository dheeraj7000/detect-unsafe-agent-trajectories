# GraphGuard — Paper Outline

**Working title:** *GraphGuard: Early Warning of Unsafe Long-Horizon Agent Behavior from Partial Execution Graphs*

**Target venues:** IEEE S&P, ACM CCS, USENIX Security, NDSS, or ICLR (ML safety track)

---

## 1. Abstract

Long-horizon autonomous AI agents are increasingly capable (METR reports frontier models completing multi-hour tasks), yet remain vulnerable to indirect prompt injection, memory poisoning, privilege escalation, and data exfiltration (AgentDojo, AgentDyn). Existing defenses (e.g., AgentArmor) enforce policies at runtime but do not address **predictive lead time** — how early unsafe behavior can be detected before the violation occurs.

We present GraphGuard, an instrumented agent-control testbed that tests whether **typed temporal behavior graphs** provide earlier, better-calibrated warning of unsafe long-horizon behavior than ordinary trajectory (sequence) classifiers. We build an Agent Control & Evaluation Lab with an MCP-compatible gateway, sandboxed tool execution, and online graph construction, then evaluate six monitor types across six attack scenarios.

At matched false-positive rate (1%/5%) and preserved benign task utility, our experiments show the graph monitor detects unsafe trajectories **earlier** than sequence baselines (lower normalized detection time, more lead actions), with the advantage persisting on longer horizons and unseen attack families. We release the gateway, graph schema, replayable scenarios, benchmark scripts, trace viewer, and trained baselines as a public artifact.

**Key finding:** Explicit provenance, authorization, and dependency structure improves time-to-detection under realistic low-FPR constraints — especially as trajectories grow longer.

---

## 2. Introduction

- Motivation: rising agent task horizons (METR) + persistent vulnerabilities (indirect prompt injection, dynamic attacks)
- Problem: sequence classifiers must infer cross-step dependencies from flat traces; unsafe behavior is often relational (a harmless tool call becomes risky when connected to untrusted input, credential, privilege change, or later external action)
- Hypothesis: a typed temporal graph exposes these dependencies explicitly, enabling earlier warning
- Contributions: (1) canonical behavior graph schema, (2) online graph monitor, (3) matched-FPR early-detection evaluation protocol, (4) reproducible Agent Control & Evaluation Lab release
- Falsifiable decision rule stated upfront

---

## 3. Related Work

- METR task-completion time horizons (frontier-agent capability)
- AgentDojo / AgentDyn (prompt injection benchmarks for tool-using agents)
- AgentArmor (structured trace graphs for policy enforcement — but no predictive lead-time study)
- LLM agent security: indirect prompt injection, memory poisoning, privilege escalation
- Graph ML for security: provenance graphs, dependency graphs

---

## 4. Method

### 4.1 Agent Control & Evaluation Lab
- Agent (LLM + planner + memory), Gateway (MCP proxy + policy engine + identity), Sandbox (Dockerized tools), Telemetry (OpenTelemetry-style events, JSONL/Parquet)
- Online behavior-graph construction alongside raw trajectories

### 4.2 Behavior Graph Schema
- Nodes: goal, subgoal, observation, instruction, tool_call, tool_result, file/resource, credential, memory write/read, policy decision, user approval, external effect
- Edges: follows, derived_from, reads, writes, authorizes, invokes, depends_on, contradicts, escalates_privilege, sends_to, persists_to_memory
- Example risk pattern: untrusted tool output → plan update → credential read → external send

### 4.3 Monitors
- Rules / policy engine (high-precision interpretable baseline)
- Feature classifier (aggregated trace statistics + logistic regression)
- Sequence classifier (TF-IDF over serialized trajectory + logistic regression) — primary baseline
- Temporal sequence model (event embeddings + MLP) — strong long-context baseline
- Graph monitor (typed temporal GNN: node/edge embeddings, K rounds of typed message passing, mean-pooling readout, sigmoid risk head)
- Hybrid (graph + sequence + rules)

### 4.4 Evaluation Protocol
- Unit of analysis: episode (20–500+ actions) with known unsafe-event onset
- At every prefix, each monitor emits a risk score; compare how early each crosses a calibrated alert threshold
- Matched-FPR comparison (1% and 5%), not matched threshold
- Metrics: normalized detection time, lead actions, TPR@FPR, AUROC/AUPRC, task utility, monitoring overhead, calibration (ECE/Brier)

---

## 5. Experiments

### 5.1 Dataset
- Benign + adversarial long-horizon episodes across 6 threat scenarios:
  1. Indirect prompt injection / tool-output hijacking
  2. Memory poisoning
  3. Privilege escalation / permission-boundary probing
  4. Sensitive-data exfiltration through legitimate tools
  5. Goal drift / specification gaming
  6. Multi-agent delegation
- Labels derived from environment state + explicit policy (not LLM judge)

### 5.2 Hypotheses (H1–H5)
- H1: Earlier warning (graph beats sequence at matched FPR)
- H2: Horizon robustness (graph degrades less as episodes grow longer)
- H3: Structural generalization (transfers to unseen attack wording)
- H4: Interpretability (risk localizes to provenance/auth violation subgraphs)
- H5: Hybrid value (graph + sequence + rules Pareto frontier)

### 5.3 Ablations
- Remove provenance / authorization / temporal edges; untype nodes/edges
- Full graph vs bounded recent subgraph
- Train on one attack family, test on unseen
- Counterfactual replay: alter one dependency (remove permission, sanitize output)

### 5.4 Results
- H1 supported: graph detects earlier (lower normalized detection time, more lead actions)
- Statistical comparison: paired t-test, Wilcoxon signed-rank, bootstrap CI for difference
- Negative-result framing: identify redundant structure and unjustified overhead

---

## 6. Discussion

- Limitations: mock sandbox/LLM in current implementation; needs real agent integration
- Negative results are informative (which graph structure is redundant)
- Ethical considerations: defensive research, responsible disclosure

---

## 7. Conclusion

- Explicit dependency structure improves early warning
- Public release enables replication and extension

---

## 8. Artifacts (Public Release)

- Gateway, graph schema, replayable scenarios, benchmark scripts
- Trace viewer / dashboard
- Trained baselines
- Research report

---

## References

- METR (2026). Task-Completion Time Horizons of Frontier AI Models.
- Debenedetti et al. (2024). AgentDojo. arXiv:2406.13352.
- Li et al. (2026). AgentDyn. arXiv:2602.03117.
- Wang et al. (2025). AgentArmor. arXiv:2508.01249.
