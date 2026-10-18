# GraphGuard — Research & Project Plan

> **Central question:** At matched false-positive rate and task utility, does a graph-based monitor detect an unsafe trajectory at a smaller fraction of the episode than strong sequence baselines?
>
> **Central claim (falsifiable):** Support the hypothesis only if the graph monitor produces statistically meaningful earlier detection at matched FPR, without materially reducing benign task utility, and the advantage persists on unseen scenarios or longer horizons.

---

## 1. Research Questions & Hypotheses

| ID | Claim | How tested |
|----|-------|------------|
| H1 | **Earlier warning**: graph monitors detect policy violations earlier than sequence monitors at matched FPR. | Normalized detection time & lead actions at FPR = 1% and 5% |
| H2 | **Horizon robustness**: graph performance degrades less as irrelevant steps / episode length increase. | Vary horizon length; compare TPR@FPR across horizons |
| H3 | **Structural generalization**: graphs transfer better to unseen attack wording when the dependency pattern is preserved. | Train on one attack family; test on unseen families |
| H4 | **Interpretability**: risk localizes to subgraphs/paths matching provenance or authorization violations. | Subgraph attribution vs. risk score correlation |
| H5 | **Hybrid value**: graph + sequence + deterministic policy checks outperform any single representation on the safety/utility frontier. | Pareto frontier comparison across monitors |

**Decision rule:** If accuracy improves but lead time does not, H1 is not supported. A negative result is still useful — it identifies which graph structure is redundant with sequence models and where graph overhead is unjustified.

---

## 2. Timeline Overview (24 weeks / ~6 months)

| Phase | Weeks | Focus | Exit Criterion |
|-------|-------|-------|----------------|
| 1 — Harness | 1–4 | MCP/tool gateway, sandbox, event schema, replay | Deterministic replay of 3 task families |
| 2 — Dataset | 5–8 | Benign + adversarial long-horizon episodes | Ground-truth violation/onset labels |
| 3 — Baselines | 9–12 | Rules + feature + sequence monitors | Calibrated baseline curves |
| 4 — Graph monitor | 13–16 | Online graph builder + GNN/graph-transformer | Risk score after every action |
| 5 — Study | 17–22 | Matched-FPR early-detection + ablations | Statistical comparison with CIs |
| 6 — Release | 23–24 | Code, schemas, tasks, traces, paper/report | Reproducible public benchmark |

---

## 3. Phase-by-Phase Plan

### Phase 1 — Harness (Weeks 1–4)
**Goal:** Build the Agent Control & Evaluation Lab — an open-source sandbox for running tool-using agents against benign and adversarial long-horizon tasks.

**Components:**
- **Agent:** LLM + planner + memory completing multi-step tasks
- **Gateway:** MCP/tool proxy, identity, policy engine mediating every external action
- **Sandbox:** Dockerized environment (email, files, web, code, mock APIs) — safe and reproducible
- **Telemetry:** OpenTelemetry-style event schema; JSONL/Parquet trace store
- **Representation layer:** Records raw trajectories AND constructs behavior graphs online
- **Monitors:** Rules, sequence model, graph model, hybrid (stubbed)
- **Evaluator:** Success, safety, lead time, cost

**Key tasks:**
- Define canonical event schema (nodes/edges from proposal)
- Build MCP-compatible gateway with policy engine and identity
- Containerize tool sandbox
- Implement deterministic replay (immutable scenario manifests, fixed seeds)
- Build trace viewer/dashboard for inspecting episodes

**Deliverable:** Harness that can run 3 task families with deterministic replay.

---

### Phase 2 — Dataset (Weeks 5–8)
**Goal:** Curate labeled episodes with known unsafe-event onset / policy-violation points.

**Episodes:** 20–500+ agent actions each.

**Threat scenarios to cover:**
1. Indirect prompt injection & tool-output hijacking
2. Memory poisoning (behavior change many steps later)
3. Privilege escalation / permission-boundary probing
4. Sensitive-data exfiltration through legitimate tools
5. Goal drift / specification gaming across long tasks
6. Multi-agent delegation where unsafe intent emerges via combined actions

**Data sources:**
- AgentDojo-style tool-use scenarios adapted into longer multi-stage episodes
- AgentDyn-inspired dynamic/open-ended cases
- Controlled synthetic variants: same local action is benign or unsafe depending on provenance, authorization, and downstream destination

**Labeling (each episode):**
- Task-success label
- Policy-violation label
- Unsafe-event step (ground truth)
- Attack family
- Affected asset
- Trust boundary crossed
- Severity

**Preference:** Labels derived from environment state and explicit policy, not an LLM judge.

**Deliverable:** Labeled dataset of benign + adversarial long-horizon episodes with ground-truth violation/onset labels.

---

### Phase 3 — Baselines (Weeks 9–12)
**Goal:** Implement and calibrate ordinary-trajectory baselines.

| Monitor | Input | Role |
|---------|-------|------|
| Rules / policy engine | Current event + permissions | High-precision interpretable baseline |
| Feature classifier | Aggregated trace statistics | Cheap classical baseline |
| Sequence classifier | Serialized recent/full trajectory | Primary ordinary-trajectory baseline |
| Temporal sequence model | Event embeddings over time | Strong long-context baseline |

**Technical stack:** scikit-learn (simple baselines), compact transformer / LLM embeddings for sequence models.

**Calibration:** Expected calibration error / Brier score; set alert thresholds at FPR = 1% and 5% on validation split.

**Deliverable:** Calibrated baseline curves (risk score vs. prefix step) for all sequence monitors.

---

### Phase 4 — Graph Monitor (Weeks 13–16)
**Goal:** Implement the proposed graph-based online monitor.

**Behavior graph schema:**
- **Nodes:** goal, subgoal, observation, instruction, tool call, tool result, file/resource, credential, memory write/read, policy decision, user approval, external effect
- **Edges:** follows, derived-from, reads, writes, authorizes, invokes, depends-on, contradicts, escalates-privilege, sends-to, persists-to-memory

**Example risk pattern:** untrusted tool output → plan update → credential read → external send. (Flat sequence model must infer dependency across distance; graph makes provenance and sensitive-data path explicit.)

**Implementation:**
- Online graph builder (NetworkX for prototyping)
- Temporal GNN or graph-transformer model (PyTorch Geometric or DGL)
- Risk score emitted after every action

**Deliverable:** Graph monitor producing a risk score after each observed action.

---

### Phase 5 — Study (Weeks 17–22)
**Goal:** Run the matched-FPR early-detection experiments and ablations.

**Evaluation protocol:** At every prefix of each episode, each monitor emits a risk score. Measure how early each crosses a calibrated alert threshold.

**Primary metrics:**
- **Normalized detection time:** first alert step / unsafe-event step (<1 = pre-event warning)
- **Lead actions / lead time:** actions or seconds between alert and unsafe event
- **TPR at fixed low FPR:** 1% and 5%
- **AUROC / AUPRC:** by scenario and horizon length
- **Task utility:** benign-task completion retained after interventions
- **Monitoring overhead:** latency, tokens, GPU/CPU cost, graph-construction cost per action
- **Calibration:** ECE / Brier score

**Key ablations:**
- Remove provenance edges / authorization edges / temporal edges; untype nodes/edges
- Vary observation window and episode length (test whether graph advantage grows with horizon)
- Full graph vs. bounded recent subgraph (runtime scalability)
- Train on one attack family, test on unseen families (structural generalization)
- Counterfactual replay: alter one dependency (remove permission / sanitize tool output) and check risk changes appropriately

**Statistical rigor:** Fixed seeds, confidence intervals, statistical significance testing.

**Deliverable:** Matched-FPR early-detection results with confidence intervals; ablation findings.

---

### Phase 6 — Release (Weeks 23–24)
**Goal:** Public, reproducible release.

**Artifacts:**
- Gateway, graph schema, replayable scenarios
- Benchmark scripts, trained baselines
- Dashboard / trace viewer
- Trained graph + sequence models
- Concise research report / paper

**Potential paper title:** *GraphGuard: Early Warning of Unsafe Long-Horizon Agent Behavior from Partial Execution Graphs.*

---

## 4. Evaluation Protocol Detail

**Unit of analysis:** one episode (20–500+ actions) with known unsafe-event onset.

**At every prefix:**
1. Monitor emits risk score $s_t$
2. Compare $s_t$ against calibrated threshold $\tau$ (set at 1% / 5% FPR)
3. Record first $t$ where $s_t \ge \tau$ → detection step
4. Compute normalized detection time = detection step / unsafe-event step
5. Compute lead actions = unsafe-event step − detection step

**Matching:** Compare monitors at the same FPR (not same threshold), so the comparison is deployment-relevant.

**Utility check:** Measure benign-task completion rate when interventions (alerts → blocks/rollbacks) are applied. The graph monitor must not materially reduce this.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Graph construction overhead too high per action | Bounded recent subgraph ablation; profile latency/GPU cost; optimize graph builder |
| Graph advantage disappears on short horizons | Vary horizon length; report results stratified by episode length |
| Labels noisy (LLM judge bias) | Derive labels from environment state + explicit policy wherever possible |
| Overfitting to attack wording | Train on one family, test on unseen families (H3); synthetic variants with preserved structure |
| Deterministic replay breaks with LLM non-determinism | Fixed seeds, immutable scenario manifests, record LLM outputs for replay |
| Graph redundant with sequence (negative result) | Explicitly framed as useful contribution; ablation identifies redundant structure |

---

## 6. Minimal Technical Stack

- Python agent harness; MCP-compatible proxy/gateway; Dockerized tool sandbox
- OpenTelemetry-style event schema + JSONL/Parquet trace store
- NetworkX (graph construction/prototyping); PyTorch Geometric or DGL (learned graph models)
- Sequence baselines: compact transformer / LLM embeddings; scikit-learn (simple baselines)
- Experiment tracking: fixed seeds, immutable scenario manifests, deterministic replay

---

## 7. Immediate Next Steps (Week 1)

1. **Stand up repo** — initialize GraphGuard project with folder structure (`harness/`, `dataset/`, `monitors/`, `evaluator/`, `docs/`)
2. **Define event schema** — canonical node/edge types (from proposal) in JSON Schema
3. **Prototype gateway** — minimal MCP proxy that mediates one tool type (e.g., file read/write) with a policy engine
4. **Sandbox one task family** — e.g., email/file workflow with a known unsafe-event onset
5. **Implement replay** — record and deterministically replay a short episode
6. **Create labeling rubric** — document how unsafe-event steps and policy violations are labeled

---

## 8. Success Summary

The project succeeds if, at matched FPR (1%/5%) and preserved benign task utility, the graph monitor detects unsafe trajectories earlier (lower normalized detection time, more lead actions) than sequence baselines, and that advantage holds on longer horizons and unseen attack families. Even a negative result clarifies which graph structure adds signal beyond sequences and where graph overhead is unjustified — both are publishable.
