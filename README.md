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

# ExamForge 🎓

> **An OpenEnv reinforcement learning environment for training LLM agents to
> generate, validate, and assemble high-quality competitive exam papers —
> targeting India's JEE, GATE, and UPSC examinations.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Space-Live-orange)](https://huggingface.co/spaces/Igris69/examforge_env)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Why ExamForge?

Each year, over **2.5 million students** appear for JEE alone. The quality of
exam questions directly shapes their preparation and outcomes. Yet current
AI tools hallucinate incorrect answers, generate trivial distractors, and
lack mechanisms to ensure paper-level balance.

ExamForge addresses this by framing exam paper creation as a **multi-step
reinforcement learning problem** — one where an agent must learn to:

1. Generate factually correct questions with non-trivial distractors
2. Validate questions using programmatic quality checks
3. Flag and remove low-quality items
4. Assemble a balanced paper meeting coverage and difficulty constraints

This is a genuine, high-stakes real-world task. Agents trained on ExamForge
learn skills that transfer directly to document quality assessment, content
moderation, and structured generation tasks.

---

## Environment Design

### Episode Flow

```
reset()
  │
  ▼
generate_question() ──► validate_question() ──► flag_question()?
  │                           │                       │
  └───────────────────────────┘ (repeat up to 50 steps)
                              │
                              ▼
                       assemble_paper() ──► done=True
```

Each episode is initialized with a randomly selected subject (JEE Physics,
JEE Chemistry, JEE Mathematics, GATE CS, or UPSC General Studies) and a
corresponding set of topics. The agent has 50 steps to build and assemble
a quality exam paper.

### Action Space

| Action | Required Fields | Purpose |
|--------|----------------|---------|
| `generate_question` | topic, difficulty, marks, question_text, option_a–d, correct_option, explanation | Inject a new MCQ into the question bank |
| `validate_question` | question_id | Score question quality programmatically |
| `flag_question` | question_id, flag_reason | Remove low-quality questions from pool |
| `assemble_paper` | *(none)* | Finalize paper and compute global score |

**Difficulty levels:** `easy` (1 mark) · `medium` (2 marks) · `hard` (4 marks)

**Subjects:** JEE Physics · JEE Chemistry · JEE Mathematics · GATE CS · UPSC GS

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | str | Unique episode identifier |
| `step_count` | int | Steps taken in current episode |
| `questions_generated` | int | Total questions in bank |
| `questions_validated` | int | Questions that passed validation |
| `questions_flagged` | int | Questions removed from pool |
| `total_marks_used` | int | Marks consumed (budget: 100) |
| `available_topics` | list[str] | Valid topics for this episode |
| `paper_constraints` | dict | `{total_marks, num_questions, time_limit_mins}` |
| `last_action_result` | str | Human-readable result of last action |
| `last_action_success` | bool | Whether last action succeeded |
| `question_id_generated` | str \| None | ID of newly created question |
| `validation_score` | float | Quality score after validate_question [0–1] |
| `paper_assembled` | bool | True when paper is finalized |
| `final_paper_score` | float | Composite paper quality score [0–1] |
| `topic_coverage_score` | float | Topic diversity score [0–1] |
| `difficulty_distribution` | dict | Counts by difficulty level |
| `reward` | float | Step reward signal |
| `done` | bool | True when episode ends |

### Reward Structure

| Action | Condition | Reward |
|--------|-----------|--------|
| `generate_question` | Well-formed, valid fields, within marks budget | **+0.30** |
| `generate_question` | Missing fields or invalid topic/difficulty/marks | **−0.20** |
| `generate_question` | Would exceed marks budget | **−0.30** |
| `validate_question` | High quality (score > 0.7) | **+0.70 to +1.00** |
| `validate_question` | Medium quality (score 0.4–0.7) | **+0.30 to +0.60** |
| `validate_question` | Low quality (score < 0.4) | **+0.00** |
| `flag_question` | Correctly flags a low-quality question | **+0.20** |
| `flag_question` | Flags a valid high-quality question | **−0.10** |
| `assemble_paper` | Complete, balanced paper (10+ valid questions) | **+1.50 to +3.00** |
| `assemble_paper` | Insufficient valid questions (< 10) | **−0.50** |
| Episode timeout | >50 steps without assembly | **−1.00** |

The reward is **dense** — the agent receives signal at every step, not just at
episode end. This enables stable RL training with short-horizon credit assignment.

### Programmatic Validation Checks

When `validate_question` is called, three deterministic checks run:

1. **Answer consistency (+0.20):** The explanation explicitly references the
   correct option letter (A, B, C, or D).
2. **Distractor quality (+0.30):** All four options are distinct, non-empty,
   and of sufficient length (>3 characters each).
3. **Difficulty calibration (+0.20):** Explanation depth matches claimed
   difficulty (hard → 150+ chars, medium → 50+ chars, easy → any).

Total maximum: 0.70. Normalized to [0.0, 1.0] for grader output.

---

## Tasks

### Task 1 — Question Generation (`easy`)

**Objective:** Generate 12+ MCQs covering ≥5 topics with diverse difficulty levels.

**Grader metrics:**
- Question count score: `min(total / 15.0, 1.0)` — weight 40%
- Topic diversity: `min(unique_topics / 5.0, 1.0)` — weight 40%
- Marks variety: `min(unique_marks / 3.0, 1.0)` — weight 20%

**Expected agent score:** 0.60–0.99 for a capable LLM

---

### Task 2 — Question Validation (`medium`)

**Objective:** Generate questions, validate all of them, and flag those with
validation score < 0.4.

**Grader metrics:**
- Validation coverage: proportion of questions validated — weight 40%
- Average quality: mean validation score — weight 40%
- Flagging precision: correctly flagged low-quality / all flagged — weight 20%

**Expected agent score:** 0.50–0.90 for a capable LLM

---

### Task 3 — Paper Assembly (`hard`)

**Objective:** Run the full pipeline — generate, validate, flag, then assemble
a balanced paper with ≥10 valid questions, good topic coverage, and the target
difficulty distribution (30% easy, 50% medium, 20% hard).

**Grader formula:**
```
score = 0.30 × assembly_bonus
      + 0.30 × topic_coverage_score
      + 0.25 × difficulty_balance_score
      + 0.15 × validation_ratio
```

**Expected agent score:** 0.40–0.85 for a capable LLM

---

## Baseline Scores

Run `python inference.py` against `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| question_generation | Easy | ~0.82 |
| question_validation | Medium | ~0.67 |
| paper_assembly | Hard | ~0.54 |

*Scores are deterministic with `temperature=0.3` and the provided sample questions.*

---

## Quick Start

### Installation

```bash
# Install the client
pip install openenv-core

# Clone the environment
git clone https://huggingface.co/spaces/Igris69/examforge_env
cd examforge_env
pip install -e .
```

### Run the Demo Agent

```bash
# Set your credentials
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run the baseline inference script
python inference.py
```

### Use the Environment Programmatically

```python
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ActionType

env = ExamForgeEnvironment()
obs = env.reset()

print(f"Subject: {env.current_subject}")
print(f"Topics: {obs.available_topics[:3]}")

# Generate a question
action = ExamForgeAction(
    action_type=ActionType.GENERATE_QUESTION,
    topic="Kinematics",
    difficulty="medium",
    marks=2,
    question_text="A particle moves with uniform acceleration. Which graph shows velocity vs time?",
    option_a="Horizontal straight line",
    option_b="Straight line with positive slope",
    option_c="Parabola",
    option_d="Vertical straight line",
    correct_option="B",
    explanation="Option B is correct. For uniform acceleration, v = u + at, which is linear in t — "
                "giving a straight line with positive slope on a velocity-time graph.",
)
result = env.step(action)
print(f"Reward: {result.reward}, Question ID: {result.question_id_generated}")
```

---

## Project Structure

```
examforge_env/
├── __init__.py              # Package exports
├── models.py                # Pydantic Action/Observation schemas (OpenEnv spec)
├── client.py                # EnvClient wrapper for remote connections
├── inference.py             # LLM agent baseline (runs all 3 tasks)
├── test_agent.py            # Unit tests for environment logic
├── openenv.yaml             # Environment manifest (OpenEnv spec)
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── Dockerfile               # Container spec (port 7860)
├── validate-submission.sh   # Official submission validator
└── server/
    ├── environment.py       # Core RL environment + 3 grader functions
    └── app.py               # FastAPI server (OpenEnv endpoints)
```

---

## Environment Specifications

| Property | Value |
|----------|-------|
| Max steps per episode | 50 |
| Marks budget | 100 |
| Target questions | 25 |
| Subjects supported | 5 (JEE Physics, Chemistry, Maths; GATE CS; UPSC GS) |
| Topics per subject | 10 |
| Reward range | [−1.0, +3.0] per step |
| Episode terminates on | `assemble_paper` success OR step limit reached |
| Concurrent sessions | 32 (configurable) |
| Runtime requirements | 2 vCPU, 4GB RAM minimum |

---

## Validation

Run the official submission validator:

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://igris69-examforge-env.hf.space .
```

Expected output:
```
PASSED -- HF Space is live and responds to /reset
PASSED -- inference.py contains correct log format
PASSED -- Docker build succeeded
PASSED -- openenv validate passed
All 4/4 checks passed!
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| RL Framework | [Meta PyTorch OpenEnv](https://github.com/meta-pytorch/OpenEnv) |
| Deployment | [HuggingFace Spaces](https://huggingface.co/spaces) (Docker) |
| API Server | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Data Models | [Pydantic v2](https://docs.pydantic.dev/) |
| LLM Client | [OpenAI SDK](https://github.com/openai/openai-python) (OpenAI-compatible) |
| Python | 3.11+ |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 — Scaler School of Technology, Bangalore.*
