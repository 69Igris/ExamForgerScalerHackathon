---
title: ExamForge
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
tags:
  - openenv
  - rl-training
  - education
  - reasoning
---

# ExamForge

> **An OpenEnv reinforcement learning environment for training LLM agents to
> generate, validate, and assemble high-quality competitive exam papers —
> targeting India's JEE, GATE, and UPSC examinations.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/HF-Space%20Live-orange)](https://huggingface.co/spaces/Igris69/examforge_env)

---

## The Problem

**2.5 million students** appear for JEE annually. The quality of exam questions
directly determines their preparation and outcomes. Yet current AI tools:

- Hallucinate incorrect answers and wrong formulas
- Generate trivial distractors that give away the answer
- Cannot balance difficulty across a complete paper
- Lack any feedback mechanism to improve question quality

No closed-loop RL environment exists for this problem.

## The Solution

ExamForge frames exam paper creation as a **multi-step reinforcement learning
problem** with three progressive tasks:

```
Task 1 (Easy):   Generate -> score on count, diversity, marks variety
Task 2 (Medium): Generate -> Validate -> Flag -> score on coverage and quality
Task 3 (Hard):   Generate -> Validate -> Flag -> Assemble -> score on balance
```

An agent trained on ExamForge learns to:
1. Produce factually correct questions with non-trivial distractors
2. Identify and remove low-quality questions programmatically
3. Balance topics and difficulty across a complete 100-mark paper

---

## Episode Flow

```
reset()
   |
   v
generate_question() --> validate_question() --> [flag_question()?]
   |                          |                        |
   +-------(repeat, max 50 steps)---------------------+
                               |
                               v
                        assemble_paper()
                               |
                               v
                          done=True
```

---

## Action Space

| Action | Required Fields | Reward |
|--------|----------------|--------|
| `generate_question` | topic, difficulty, marks, question_text, option_a-d, correct_option, explanation | +0.30 valid / -0.20 invalid |
| `validate_question` | question_id | +0.0 to +1.0 (by quality) |
| `flag_question` | question_id, flag_reason | +0.20 (correct) / -0.10 (incorrect) |
| `assemble_paper` | *(none)* | +1.50 to +3.00 (by balance) |

**Subjects:** JEE Physics, JEE Chemistry, JEE Mathematics, GATE CS, UPSC GS

**Difficulty levels:** easy (1 mark), medium (2 marks), hard (4 marks)

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | str | Unique episode identifier |
| `step_count` | int | Steps used (max: 50) |
| `available_topics` | list[str] | Valid topics for this episode |
| `paper_constraints` | dict | `{total_marks: 100, num_questions: 25}` |
| `questions_generated` | int | Questions in bank |
| `questions_validated` | int | Questions that passed validation |
| `total_marks_used` | int | Marks consumed (budget: 100) |
| `last_action_result` | str | Human-readable result of last action |
| `last_action_success` | bool | Whether action succeeded |
| `question_id_generated` | str | ID of newly created question |
| `validation_score` | float | Quality score [0.0-1.0] |
| `paper_assembled` | bool | True after assemble_paper succeeds |
| `final_paper_score` | float | Overall paper quality [0.0-1.0] |
| `topic_coverage_score` | float | Topic diversity score |
| `difficulty_distribution` | dict | Count by difficulty level |
| `available_actions` | list[str] | Valid actions at current step |
| `reward` | float | Step reward signal |
| `done` | bool | Episode terminal flag |

---

## Reward Shaping

Dense rewards at every step, not just at episode end:

| Action | Condition | Reward |
|--------|-----------|--------|
| `generate_question` | Well-formed, valid topic, within marks budget | **+0.30** |
| `generate_question` | Missing fields / invalid topic / marks violation | **-0.20 to -0.30** |
| `validate_question` | High quality (score > 0.7) | **+0.70 to +1.00** |
| `validate_question` | Medium quality (0.4-0.7) | **+0.30 to +0.60** |
| `validate_question` | Low quality (< 0.4) | **+0.00** |
| `flag_question` | Correctly flags low-quality question | **+0.20** |
| `flag_question` | Incorrectly flags good question | **-0.10** |
| `assemble_paper` | Complete, balanced paper (10+ valid questions) | **+1.50 to +3.00** |
| `assemble_paper` | Insufficient questions | **-0.50** |
| Episode timeout | Step limit reached without assembly | **-1.00** |

### Programmatic Validation (3 deterministic checks):
1. **Answer consistency (+0.20):** Explanation explicitly references correct option letter
2. **Distractor quality (+0.30):** All 4 options distinct, non-empty, plausible
3. **Difficulty calibration (+0.20):** Explanation depth matches claimed difficulty

---

## Tasks

### Task 1 — Question Generation (Easy)
Generate 12+ MCQs covering 5+ topics with diverse difficulty and marks.

**Grader formula:** `0.4 * count_score + 0.4 * topic_diversity + 0.2 * marks_variety`

Expected LLM score: **0.60-0.99**

---

### Task 2 — Question Validation (Medium)
Generate questions, validate all, flag those with quality score < 0.4.

**Grader formula:** `0.4 * validation_coverage + 0.4 * avg_quality + 0.2 * flagging_precision`

Expected LLM score: **0.50-0.90**

---

### Task 3 — Paper Assembly (Hard)
Complete pipeline: generate, validate, flag, assemble a 100-mark balanced paper.

**Grader formula:** `0.30 * assembly_bonus + 0.30 * topic_coverage + 0.25 * difficulty_balance + 0.15 * validation_ratio`

**Target distribution:** 30% easy / 50% medium / 20% hard

Expected LLM score: **0.40-0.85**

---

## Baseline Scores

Deterministic policy (no LLM required):

| Task | Deterministic Score |
|------|---------------------|
| question_generation | ~0.92 |
| question_validation | ~0.90 |
| paper_assembly | ~0.99 |

---

## Quick Start

```bash
# Install
pip install openenv-core
git clone https://huggingface.co/spaces/Igris69/examforge_env
cd examforge_env
pip install -e .

# Run baseline (no LLM needed)
python inference.py

# Run with real LLM
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Programmatic Usage

```python
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ActionType

env = ExamForgeEnvironment()
obs = env.reset()
print(f"Subject: {env.current_subject}")
print(f"Topics: {obs.available_topics[:3]}")

# Generate a question
result = env.step(ExamForgeAction(
    action_type=ActionType.GENERATE_QUESTION,
    topic="Kinematics", difficulty="medium", marks=2,
    question_text="A projectile is launched at 45 deg with speed 20 m/s. Range = ?",
    option_a="20 m", option_b="40 m", option_c="20 sqrt(2) m", option_d="40 sqrt(2) m",
    correct_option="B",
    explanation="Range R = u^2 sin(2theta)/g = 400 sin90/10 = 40 m. Option B is correct."
))
print(f"Reward: {result.reward:.2f}, QID: {result.question_id_generated}")
```

---

## Project Structure

```
examforge_env/
|-- inference.py              # LLM agent + deterministic baseline
|-- models.py                 # Typed Pydantic schemas (OpenEnv spec)
|-- client.py                 # Remote environment client
|-- openenv.yaml              # Environment manifest
|-- test_agent.py             # Environment unit tests (10 tests)
|-- test_inference_logging.py # Log format compliance tests (8 tests)
|-- Dockerfile                # Container (port 7860)
+-- server/
    |-- environment.py        # Core RL logic + 3 grader functions
    +-- app.py                # FastAPI: /reset /step /state /tasks /grader /health
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute action |
| GET | `/state` | Get episode state |
| GET | `/tasks` | List task catalog |
| POST | `/grader` | Grade an episode |
| GET | `/health` | Health check |
| GET | `/docs` | API documentation |

---

## License

MIT

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Scaler School of Technology, Bangalore.*
