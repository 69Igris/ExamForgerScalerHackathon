---
title: ExamForge
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ExamForge 🎓

> *An RL environment for training LLM agents to generate, validate, and assemble high-quality competitive exam papers — targeting India's JEE, GATE, and UPSC.*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-llama/openenv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-orange)](https://huggingface.co)

---

## 🎯 Why ExamForge?

India's competitive exams — **JEE**, **GATE**, and **UPSC** — collectively determine the academic and professional futures of **millions of students** every year:

- **1.5 million** students appear for JEE annually
- **800,000+** candidates take GATE every year
- **1 million+** aspirants register for UPSC Prelims

**The problem**: Current AI tools can generate questions, but they frequently **hallucinate incorrect answers**, produce **low-quality distractors**, and fail to create **balanced exam papers**. There is no closed-loop feedback system to improve generation quality.

**ExamForge solves this** by creating a **Reinforcement Learning environment** where LLM agents learn through structured rewards to:

1. ✅ Generate factually correct MCQs with high-quality distractors
2. ✅ Self-validate their own questions via programmatic graders
3. ✅ Flag and remove low-quality questions
4. ✅ Assemble balanced, topic-diverse exam papers

This creates a **closed-loop feedback system** where agents **learn from validation failures**, progressively improving question quality across episodes.

---

## 🏗️ Environment Design

### Episode Lifecycle

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
reset() ──► generate_question ──► validate_question ──► [flag_question]
                                                               │
                                                               ▼
                                                        assemble_paper ──► done ✓
```

An agent starts each episode by discovering a **randomly assigned subject** and its available topics. It then iteratively generates questions, validates them for quality, optionally flags poor ones, and finally assembles a complete exam paper. The episode ends either on successful assembly or after 50 steps.

### Actions

| Action | Required Fields | Purpose |
|--------|----------------|---------|
| `generate_question` | topic, difficulty, marks, question_text, options A-D, correct_option, explanation | Submit a new MCQ to the question bank |
| `validate_question` | question_id | Run programmatic quality checks on a question |
| `flag_question` | question_id, flag_reason | Remove a low-quality question from the valid pool |
| `assemble_paper` | *(none)* | Finalize the exam paper and compute the final score |

### Reward Structure

| Action | Condition | Reward |
|--------|-----------|--------|
| `generate_question` | Well-formed question submitted | **+0.3** |
| `generate_question` | Missing fields / invalid topic | **-0.2** |
| `generate_question` | Marks would exceed paper limit | **-0.3** |
| `validate_question` | High quality (score > 0.7) | **+0.7 to +1.0** |
| `validate_question` | Medium quality (0.4–0.7) | **+0.3 to +0.6** |
| `validate_question` | Low quality (< 0.4) | **0.0** |
| `flag_question` | Correctly flagging low-quality | **+0.2** |
| `flag_question` | Flagging a valid question | **-0.1** |
| `assemble_paper` | Complete, balanced paper | **+1.5 to +3.0** |
| `assemble_paper` | Insufficient questions (< 10) | **-0.5** |
| Episode timeout | 50 steps without assembly | **-1.0** |

### Validation Checks (Programmatic Grader)

Each `validate_question` runs three automated checks:

1. **Answer Consistency** (+0.2): Does the explanation reference the correct option?
2. **Distractor Quality** (+0.3): Are all 4 options unique and non-trivial (length > 3)?
3. **Difficulty Calibration** (+0.2): Does explanation length match difficulty level?

### Subjects & Topics

| Subject | Topics |
|---------|--------|
| **JEE Physics** | Kinematics, Laws of Motion, Work & Energy, Rotational Motion, Thermodynamics, Electrostatics, Current Electricity, Magnetism, Optics, Modern Physics |
| **JEE Chemistry** | Atomic Structure, Chemical Bonding, Thermochemistry, Electrochemistry, Organic Reactions, Coordination Chemistry, p-Block Elements, d-Block Elements, Polymers, Biomolecules |
| **JEE Mathematics** | Sets & Relations, Complex Numbers, Matrices, Calculus - Limits, Calculus - Integration, Probability, Vectors, 3D Geometry, Differential Equations, Statistics |
| **GATE CS** | Data Structures, Algorithms, Operating Systems, DBMS, Computer Networks, Theory of Computation, Compiler Design, Digital Logic, Computer Organization, Software Engineering |
| **UPSC GS** | Indian Polity, Indian Economy, Modern History, Geography, Science & Technology, Environment & Ecology, International Relations, Art & Culture, Ancient History, Social Issues |

---

## 🚀 Quick Start

### Installation

```bash
pip install openenv-core
git clone https://huggingface.co/spaces/Igris69/examforge_env
cd examforge_env
pip install -e .
```

### Basic Usage (Sync API)

```python
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ActionType

env = ExamForgeEnvironment()
obs = env.reset()
print(f"Subject: {env.current_subject}")
print(f"Topics: {obs.available_topics}")

# Generate a question
obs = env.step(ExamForgeAction(
    action_type=ActionType.GENERATE_QUESTION,
    topic=obs.available_topics[0],
    difficulty="medium", marks=2,
    question_text="What is Newton's second law?",
    option_a="F = ma", option_b="F = mv",
    option_c="F = m/a", option_d="F = a/m",
    correct_option="A",
    explanation="Newton's second law states F = ma. Option A is correct."
))
print(f"Reward: {obs.reward}, Question ID: {obs.question_id_generated}")
```

### Running the Demo Agent

```bash
python inference.py
```

This runs a complete episode with 15 pre-written JEE Physics MCQs, demonstrating the full generate → validate → assemble lifecycle.

---

## 🧠 Training Potential

An LLM agent trained on ExamForge would learn to:

| Capability | How ExamForge Teaches It |
|-----------|------------------------|
| **Factual correctness** | Negative rewards for inconsistent answer-explanation pairs |
| **High-quality distractors** | Validation checks penalize trivial or duplicate options |
| **Difficulty calibration** | Reward shaping based on explanation depth vs. stated difficulty |
| **Topic diversity** | Paper assembly score rewards coverage across multiple topics |
| **Paper balancing** | Final score penalizes skewed difficulty distributions |
| **Generalization** | Random subject assignment per episode forces multi-domain learning |

### Why Reward Shaping Matters

ExamForge uses **multi-step reward shaping** rather than a single end-of-episode reward. This provides:
- **Dense feedback** at every step, accelerating learning
- **Decomposed signals** that teach specific skills (quality vs. coverage vs. balance)
- **Curriculum learning** — agents naturally progress from generating any question to generating high-quality, balanced papers

---

## 📊 Environment Specs

| Property | Value |
|----------|-------|
| Max Steps per Episode | 50 |
| Target Paper Marks | 100 |
| Target Questions | 25 |
| Supported Subjects | 5 (JEE Physics, JEE Chemistry, JEE Maths, GATE CS, UPSC GS) |
| Reward Range | -1.0 to +3.0 per step |
| Evaluation | Programmatic grader + Rubric-based scoring |
| Paper Score Formula | 0.4 × topic_coverage + 0.4 × difficulty_balance + 0.2 × validation_ratio |
| Server Protocol | FastAPI + WebSocket (OpenEnv standard) |
| Deployment | Docker on HuggingFace Spaces (port 7860) |

---

## 📁 Project Structure

```
examforge_env/
├── __init__.py              # Exports: ExamForgeAction, ExamForgeObservation, ExamForgeEnv
├── models.py                # Pydantic Action & Observation models
├── client.py                # ExamForgeEnv(EnvClient) — client-side connector
├── openenv.yaml             # Environment manifest for HuggingFace Hub
├── README.md                # This file
├── pyproject.toml           # Package configuration
├── requirements.txt         # Development dependencies
├── inference.py             # Demo agent — full episode walkthrough
├── test_agent.py            # Unit tests (5 tests)
├── validate-submission.sh   # Submission validation script
├── .gitignore               # Standard Python gitignore
├── .env                     # Environment variables template
├── AGENT_INSTRUCTIONS.md    # Complete specification document
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI app (openenv create_app)
    ├── environment.py       # ExamForgeEnvironment — core RL logic
    ├── Dockerfile           # Docker container for HF Spaces
    └── requirements.txt     # Server-side dependencies
```

---

## 🤝 Built With

- **[Meta PyTorch OpenEnv Framework](https://github.com/meta-llama/openenv)** — Standardized RL environment protocol
- **[HuggingFace Spaces](https://huggingface.co/spaces)** — Cloud deployment
- **[FastAPI](https://fastapi.tiangolo.com/)** + **Docker** — Server infrastructure
- **[Pydantic](https://docs.pydantic.dev/)** — Type-safe data models

---

## 🧪 Testing

```bash
# Run unit tests
python -m pytest test_agent.py -v

# Run demo agent
python inference.py

# Validate submission
bash validate-submission.sh
```

---

## 📄 License

MIT

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Scaler School of Technology*
