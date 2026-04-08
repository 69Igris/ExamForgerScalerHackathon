---
title: ExamForge
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ExamForge

> An advanced Reinforcement Learning environment designed for training Large Language Models to autonomously generate, programmatically validate, and optimally assemble competitive examination papers, specifically targeting India's leading entrance tests: JEE, GATE, and UPSC.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-llama/openenv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Spaces-orange)](https://huggingface.co)

---

## Abstract and Problem Statement

The integrity and quality of high-stakes competitive examinations in India—namely **JEE**, **GATE**, and **UPSC**—are foundational to the country's academic and professional ecosystem. Each year, these exams dictate the career trajectories of over three million candidates.

**The Paradigm Gap**: Current generative AI tools exhibit significant shortcomings in educational assessment creation. They frequently hallucinate distractors, fail to calibrate difficulty, and lack the macro-level intelligence required to assemble a balanced paper. Traditional models lack a closed-loop feedback mechanism, leading to compounding degradation in assessment quality.

**The ExamForge Solution**: ExamForge bridges this gap by abstracting the pedagogical assessment lifecycle into a robust Reinforcement Learning environment. Utilizing the OpenEnv framework, it enables LLM agents to iteratively learn through a highly structured, multi-modal reward system, mastering the art of:
1. Synthesizing factually rigorous Multiple Choice Questions (MCQs) with high-fidelity distractors.
2. Subjecting generated items to strict programmatic quality assurance constraints.
3. Intelligently pruning suboptimal candidates from the active state pool.
4. Architecting a comprehensively balanced, topic-diverse final examination paper.

---

## Architectural Design

ExamForge employs a continuous feedback architecture conceptually mapped as a Directed Acyclic Graph during episode traversal.

### Epistemic Lifecycle

```text
    [Initialization]
           │
           ▼
  [Question Generation] ◄──────────────┐
           │                           │
           ▼                           │
 [Programmatic Validation]             │
           │                           │
           ├──► [Quality Flagging] ────┤ (Iterative Refinement)
           │                           │
           ▼                           │
   [Paper Assembly] ───────────────────┘
           │
           ▼
     [Termination]
```

At the onset of each episode, the agent is instantiated within a dynamic context featuring a randomly assigned academic discipline and its corresponding taxonomic hierarchy. The agent initiates an iterative process of generation and strict validation, progressively building a high-fidelity internal state (the question bank).

### Dimensional Actions Space

| Action Primitive | Payload Vector | Systemic Function |
|--------|----------------|---------|
| `generate_question` | `topic, difficulty, marks, question_text, options, correct_option, explanation` | Injects a generated MCQ proposition into the active memory pool. |
| `validate_question` | `question_id` | Triggers a deterministic, algorithmic evaluation suite across predefined heuristic rubrics. |
| `flag_question` | `question_id, flag_reason` | Executes a deletion operator on the internal pool, isolating structurally weak items. |
| `assemble_paper` | *(Null)* | Computes the global gradient score based on combinatorial balance and task saturation. |

---

## Pedagogical Reward Shaping

ExamForge uses a dense, gradient-rich reward topology to enforce strict convergence toward educational excellence, shifting away from sparse episodic rewards.

| Action State | Transition Condition | Reward Scalar |
|--------|-----------|--------|
| `generate_question` | Structurally absolute, syntactically parsed formulation | **+0.3** |
| `generate_question` | Dimensional constraint violation (e.g., marks out of bounds) | **-0.3** |
| `validate_question` | Exceptional heuristic resonance (Score > 0.7) | **+0.7 to +1.0** |
| `validate_question` | Marginal algorithmic compliance (Score 0.4-0.7) | **+0.3 to +0.6** |
| `flag_question` | Precision isolation of low-quality variables | **+0.2** |
| `assemble_paper` | Global optimum achieved (coverage, balance, fidelity) | **+1.5 to +3.0** |
| Timeout Condition | Episode starvation (>50 uncommitted steps) | **-1.0** |

### Algorithmic Validation Subroutine

Each validation call computes internal consistency through exact heuristics:
1. **Semantic Corroboration** (+0.2): Ensures the provided rationale deterministically points to the flagged valid option.
2. **Distractor Variance** (+0.3): Computes string entropy to guarantee mutual exclusivity across alternative choices.
3. **Calibrated Depth** (+0.2): Measures logical depth mapped strictly to the claimed difficulty integer.

---

## Epistemic Coverage

| Global Subject Matrix | Granular Taxonomic Nodes |
|---------|--------|
| **JEE Physics** | Kinematics, Thermodynamics, Electrostatics, Magnetism, Quantum Physics, Optics, ... |
| **GATE CompSci** | Automata Theory, Algorithmic Analysis, OS, Database Systems, Network Architecture, ... |
| **UPSC Gen. Studies** | Geopolitics, Indian Economic Policy, Environmental Ecology, Constitutional Law, ... |

---

## Environment Specifications

| Environment Parameter | Mathematical/Systemic Value |
|----------|-------|
| Maximum Temporal Horizon (Steps) | 50 |
| Constrained Paper Volume (Marks) | 100 |
| Optimal Item Count | 25 Questions |
| Global Reward Bounds | [-1.0, 3.0] per action |
| Terminal State Calculation | $0.4(\text{TopicCov}) + 0.4(\text{DiffBalance}) + 0.2(\text{ValidationRatio})$ |
| Communication Protocol | FastAPI + WebSocket (OpenEnv V1 Standard) |

---

## Technical Deployment & Quick Start

### Subsystem Initialization

```bash
pip install openenv-core
git clone https://huggingface.co/spaces/Igris69/examforge_env
cd examforge_env
pip install -e .
```

### Simulated Inference Protocol

Execute the autonomous inference engine to trace a full operational episode without invoking the Live WebSocket client.

```bash
python inference.py
```
*Note: This command natively verifies generation, programmatic validation, and the final deterministic assembly subroutines.*

---

## Comprehensive Project Topology

```text
examforge_env/
  ├── __init__.py              # Core Module Exports
  ├── models.py                # Pydantic Action/Observation Schemas
  ├── client.py                # Wrapper for OpenEnv API Interface
  ├── openenv.yaml             # Deployment & Manifest Specifications
  ├── pyproject.toml           # Core Package Management
  ├── inference.py             # Agent Logic and Episode Demonstrator
  ├── test_agent.py            # Unit Evaluation & Regression Test Suite
  └── server/
      ├── environment.py       # Core Multi-Modal RL Engine & Deterministic Graders
      ├── app.py               # REST/WebSocket Server Initiator
      └── Dockerfile           # High-Performance Containerization Spec
```

---

## Technology Stack

- **[Meta PyTorch OpenEnv](https://github.com/meta-llama/openenv)**: Foundational specification for interactive RL.
- **[HuggingFace Hub](https://huggingface.co/spaces)**: Scaleable model deployment via Docker Spaces.
- **[FastAPI](https://fastapi.tiangolo.com/)**: Asynchronous backend infrastructure.
- **[Pydantic](https://docs.pydantic.dev/)**: Strict data schema validation and runtime type coercion.

---

## License Rights

Released under the **MIT License**.

---

*Architected for the Meta PyTorch OpenEnv Hackathon 2026 — Scaler School of Technology*
