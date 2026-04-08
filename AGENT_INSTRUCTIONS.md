# AGENT INSTRUCTIONS — ExamForge OpenEnv Environment
# Meta PyTorch OpenEnv Hackathon 2026 | Scaler School of Technology

---

## PROJECT OVERVIEW

You are building **ExamForge** — a Reinforcement Learning environment built on the
Meta + HuggingFace **OpenEnv** framework. This environment trains LLM agents to
generate, validate, and assemble high-quality competitive exam questions (JEE/GATE/UPSC style).

This is a **Round 1 hackathon submission** for the Meta PyTorch OpenEnv Hackathon 2026
organized by Scaler School of Technology, sponsored by Meta, PyTorch, and HuggingFace.

---

## WHAT IS OPENENV (CRITICAL CONTEXT — READ CAREFULLY)

OpenEnv is an open-source framework by Meta & HuggingFace for creating standardized,
isolated, and reusable environments for training and deploying AI agents. Think of it
as the "universal language" for AI training environments.

### Core API (3 methods only — never deviate from this):
```python
env.reset()   # Start a new episode, return initial Observation
env.step(action)  # Execute an Action, return Observation with reward
env.state()   # Return episode metadata (step count, episode_id, etc.)
```

### Architecture:
- **Server side**: A FastAPI server running inside Docker, implementing Environment logic
- **Client side**: An EnvClient class that connects to the server via WebSocket
- **Models**: Pydantic dataclasses for Action, Observation, State (type-safe)
- **Deployment**: HuggingFace Spaces via `openenv push`

### Key imports from openenv-core:
```python
from openenv.core import Environment, EnvClient, Action, Observation, State, StepResult
```

---

## EXACT FILE STRUCTURE TO CREATE

```
examforge_env/
├── __init__.py              # Exports: ExamForgeAction, ExamForgeObservation, ExamForgeEnv
├── models.py                # Pydantic Action & Observation dataclasses
├── client.py                # ExamForgeEnv(EnvClient) — client-side connector
├── openenv.yaml             # Environment manifest for HuggingFace Hub
├── README.md                # Project documentation (judges read this first!)
├── pyproject.toml           # Package dependencies
├── requirements.txt         # Top-level pip requirements
├── inference.py             # Demo agent that solves the environment (for testing)
├── test_agent.py            # Unit tests for the environment
├── validate-submission.sh   # Shell script to validate the submission
├── .gitignore               # Standard Python gitignore
├── .env                     # Environment variables template (no secrets)
├── AGENT_INSTRUCTIONS.md    # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI app creation + server entrypoint
    ├── environment.py       # ExamForgeEnvironment(Environment) — core logic
    ├── Dockerfile           # Docker container definition
    └── requirements.txt     # Server-side pip requirements for Docker
```

---

## MODELS.PY — EXACT SPECIFICATION

### Action Types (agent can take these actions):
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal
from openenv.core import Action, Observation

class ActionType(str, Enum):
    GENERATE_QUESTION = "generate_question"
    VALIDATE_QUESTION = "validate_question"
    FLAG_QUESTION = "flag_question"
    ASSEMBLE_PAPER = "assemble_paper"

@dataclass
class ExamForgeAction(Action):
    action_type: ActionType
    
    # For generate_question:
    topic: Optional[str] = None          # e.g. "Kinematics", "Thermodynamics"
    difficulty: Optional[str] = None     # "easy" | "medium" | "hard"
    marks: Optional[int] = None          # 1, 2, or 4
    question_text: Optional[str] = None  # The MCQ question
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None
    option_d: Optional[str] = None
    correct_option: Optional[str] = None # "A" | "B" | "C" | "D"
    explanation: Optional[str] = None    # Why this answer is correct
    
    # For validate_question / flag_question:
    question_id: Optional[str] = None
    flag_reason: Optional[str] = None    # Why flagging this question
    
    # For assemble_paper: no extra fields needed
```

### Observation (what agent sees after each step):
```python
@dataclass 
class ExamForgeObservation(Observation):
    # Episode context
    episode_id: str = ""
    step_count: int = 0
    
    # Current paper state
    questions_generated: int = 0
    questions_validated: int = 0
    questions_flagged: int = 0
    total_marks_used: int = 0
    target_total_marks: int = 100
    
    # Last action result
    last_action_result: str = ""         # Human-readable result of last action
    last_action_success: bool = False
    question_id_generated: Optional[str] = None  # ID of newly generated question
    
    # Validation feedback (populated after validate_question)
    validation_score: float = 0.0        # 0.0 to 1.0
    distractor_quality_score: float = 0.0
    difficulty_accuracy: bool = False
    
    # Paper assembly feedback (populated after assemble_paper)
    paper_assembled: bool = False
    topic_coverage_score: float = 0.0
    difficulty_distribution: dict = field(default_factory=dict)
    final_paper_score: float = 0.0
    
    # Reward (standard OpenEnv field via parent Observation)
    reward: float = 0.0
    done: bool = False
    
    # Topics available in this episode
    available_topics: list = field(default_factory=list)
    paper_constraints: dict = field(default_factory=dict)
```

---

## ENVIRONMENT.PY — CORE LOGIC SPECIFICATION

### Class: ExamForgeEnvironment(Environment)

#### Episode Setup (reset method):
- Choose a random subject: "JEE Physics", "JEE Chemistry", "JEE Mathematics", "GATE CS", "UPSC GS"
- Set paper constraints: {total_marks: 100, num_questions: 25, time_limit_mins: 180}
- Provide list of available topics for that subject (hardcoded dict of 8-10 topics per subject)
- Initialize empty question bank for this episode
- Return ExamForgeObservation with available_topics and paper_constraints

#### Question Bank (hardcoded topics per subject):
```python
SUBJECT_TOPICS = {
    "JEE Physics": ["Kinematics", "Laws of Motion", "Work & Energy", 
                    "Rotational Motion", "Thermodynamics", "Electrostatics",
                    "Current Electricity", "Magnetism", "Optics", "Modern Physics"],
    "JEE Chemistry": ["Atomic Structure", "Chemical Bonding", "Thermochemistry",
                      "Electrochemistry", "Organic Reactions", "Coordination Chemistry",
                      "p-Block Elements", "d-Block Elements", "Polymers", "Biomolecules"],
    "JEE Mathematics": ["Sets & Relations", "Complex Numbers", "Matrices",
                        "Calculus - Limits", "Calculus - Integration", "Probability",
                        "Vectors", "3D Geometry", "Differential Equations", "Statistics"],
    "GATE CS": ["Data Structures", "Algorithms", "Operating Systems", "DBMS",
                "Computer Networks", "Theory of Computation", "Compiler Design",
                "Digital Logic", "Computer Organization", "Software Engineering"],
    "UPSC GS": ["Indian Polity", "Indian Economy", "Modern History", "Geography",
                "Science & Technology", "Environment & Ecology", "International Relations",
                "Art & Culture", "Ancient History", "Social Issues"]
}
```

#### Step Method Logic:

**GENERATE_QUESTION action:**
- Validate: topic must be in available_topics, difficulty must be easy/medium/hard, marks must be 1/2/4
- Validate: question_text, options A-D, correct_option, explanation must all be non-empty
- Validate: marks usage won't exceed total_marks constraint
- If all valid: assign a UUID question_id, store in episode question bank
- Reward: +0.3 base reward for a well-formed generated question
- Return observation with question_id_generated set

**VALIDATE_QUESTION action:**
- Check question_id exists in question bank
- Run programmatic checks:
  1. Answer consistency: Does the explanation mention the correct option? (+0.2)
  2. Distractor quality: Are all 4 options non-trivially different? (length check + simple heuristic) (+0.3)
  3. Difficulty calibration: "hard" questions should have longer explanations, "easy" shorter (simple proxy) (+0.2)
- Compute validation_score as weighted sum
- Reward: validation_score * 1.0
- If validation_score < 0.4: mark question as low-quality

**FLAG_QUESTION action:**
- Check question_id exists
- Check flag_reason is non-empty and > 10 characters
- Mark question as flagged (remove from valid pool)
- Reward: +0.2 (correctly identifying a problem) if question was previously low-quality, else -0.1

**ASSEMBLE_PAPER action:**
- Check at least 10 valid (non-flagged) questions in bank
- Check marks constraint is met (within 10% of target)
- Compute topic_coverage_score: unique topics / min(available_topics, 5) 
- Compute difficulty_distribution score: should be ~30% easy, 50% medium, 20% hard
- Compute final_paper_score = 0.4 * topic_coverage_score + 0.4 * difficulty_distribution_score + 0.2 * (validated_questions / total_questions)
- Reward: final_paper_score * 3.0 (big bonus for completing paper)
- Set done = True
- If < 10 questions: reward = -0.5, done = False

#### Max steps per episode: 50
#### Episode ends when: assemble_paper succeeds, or step_count >= 50

---

## APP.PY — FASTAPI SERVER

```python
from openenv.core.env_server import create_app
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ExamForgeObservation

def create_examforge_app():
    env = ExamForgeEnvironment()
    app = create_app(
        create_environment=lambda: ExamForgeEnvironment(),
        action_class=ExamForgeAction,
        observation_class=ExamForgeObservation,
        max_concurrent_envs=32,
    )
    return app

app = create_examforge_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

---

## CLIENT.PY — EXAMFORGE CLIENT

```python
from openenv.core import EnvClient
from models import ExamForgeAction, ExamForgeObservation

class ExamForgeEnv(EnvClient):
    action_class = ExamForgeAction
    observation_class = ExamForgeObservation
    
    def generate_question(self, topic, difficulty, marks, question_text, 
                          option_a, option_b, option_c, option_d, 
                          correct_option, explanation):
        """Convenience method to generate a question."""
        return self.step(ExamForgeAction(
            action_type="generate_question",
            topic=topic, difficulty=difficulty, marks=marks,
            question_text=question_text, option_a=option_a,
            option_b=option_b, option_c=option_c, option_d=option_d,
            correct_option=correct_option, explanation=explanation
        ))
    
    def validate_question(self, question_id: str):
        """Convenience method to validate a question."""
        return self.step(ExamForgeAction(
            action_type="validate_question",
            question_id=question_id
        ))
    
    def assemble_paper(self):
        """Convenience method to finalize and assemble the paper."""
        return self.step(ExamForgeAction(action_type="assemble_paper"))
```

---

## INFERENCE.PY — DEMO AGENT

This file demonstrates an agent completing the ExamForge environment.
The agent should:
1. Connect to the environment (use localhost:7860 for local testing)
2. Call reset() to get available_topics and paper_constraints
3. Generate 12-15 questions across different topics
4. Validate each generated question
5. Flag any question with validation_score < 0.4
6. Call assemble_paper() to finalize

The agent should use a simple greedy strategy — no LLM required for the demo.
Pre-write 15 sample JEE Physics MCQ questions as hardcoded strings in the script.
Show the total reward accumulated and the final paper score.

---

## DOCKERFILE — SERVER

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy server requirements
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Install the package itself
RUN pip install -e .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## SERVER/REQUIREMENTS.TXT

```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
uuid
```

---

## OPENENV.YAML — MANIFEST

```yaml
name: examforge_env
version: "1.0.0"
description: >
  ExamForge: An RL environment for training LLM agents to generate, validate,
  and assemble high-quality competitive exam papers. Targets JEE, GATE, and 
  UPSC domains with multi-step reward shaping and programmatic graders.
author: "Team ExamForge"
tags:
  - education
  - reasoning
  - question-generation
  - india
  - jee
  - gate
  - upsc
  - rl-training
action_class: ExamForgeAction
observation_class: ExamForgeObservation
max_episode_steps: 50
```

---

## PYPROJECT.TOML

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "examforge-env"
version = "1.0.0"
description = "ExamForge: Competitive Exam RL Environment for OpenEnv"
requires-python = ">=3.11"
dependencies = [
    "openenv-core>=0.2.1",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]

[tool.setuptools.packages.find]
where = ["."]
include = ["examforge_env*"]
```

---

## README.MD — WHAT JUDGES WILL READ FIRST

The README must contain:
1. **One-line pitch**: "ExamForge trains LLM agents to generate, validate, and assemble high-quality competitive exam papers for India's most competitive exams — JEE, GATE, and UPSC."
2. **Why this matters**: Explain the real-world impact (millions of students, quality exam content)
3. **Environment design**: Actions, Observations, Reward structure table
4. **Quick start**: pip install + 5-line usage example
5. **Reward shaping explanation**: Why each reward signal was chosen
6. **Topics supported**: Full list of subjects and topics
7. **Episode lifecycle**: Step-by-step walkthrough of a full episode
8. **Training potential**: How an LLM trained on this env would improve

---

## VALIDATE-SUBMISSION.SH

```bash
#!/bin/bash
echo "=== ExamForge Submission Validator ==="
echo "Checking Python version..."
python3 --version

echo "Installing dependencies..."
pip install -e . -q

echo "Running environment smoke test..."
python3 -c "
from models import ExamForgeAction, ExamForgeObservation
from server.environment import ExamForgeEnvironment
env = ExamForgeEnvironment()
obs = env.reset()
print('reset() OK — Subject:', obs.available_topics[:3])
action = ExamForgeAction(action_type='assemble_paper')
result = env.step(action)
print('step() OK — reward:', result.reward)
print('state() OK —', env.state())
print('ALL CHECKS PASSED')
"
echo "=== Validation Complete ==="
```

---

## CRITICAL RULES — DO NOT VIOLATE

1. **Never use `gym` or `gymnasium` imports directly** — only use `openenv.core`
2. **Reward must always be a float** — never None or string
3. **done=True must be set when episode ends** — either paper assembled or max steps reached
4. **All Action fields must be Optional** — agent may not always provide all fields
5. **Observation must always include step_count and episode_id**
6. **Server must bind to port 7860** — this is what HuggingFace Spaces expects
7. **No external API calls in environment logic** — grader must be self-contained
8. **Docker image must build successfully** — test with `docker build` before submitting
9. **__init__.py must export**: `ExamForgeAction`, `ExamForgeObservation`, `ExamForgeEnv`
10. **openenv.yaml must be present** — this is how the Hub indexes the environment

---

## REWARD SUMMARY TABLE

| Action | Condition | Reward |
|--------|-----------|--------|
| generate_question | Well-formed question submitted | +0.3 |
| generate_question | Missing fields / invalid topic | -0.2 |
| generate_question | Marks would exceed paper limit | -0.3 |
| validate_question | High quality (score > 0.7) | +0.7 to +1.0 |
| validate_question | Medium quality (0.4-0.7) | +0.3 to +0.6 |
| validate_question | Low quality (< 0.4) | 0.0 |
| flag_question | Correctly flagging low-quality | +0.2 |
| flag_question | Flagging a valid question | -0.1 |
| assemble_paper | Complete, balanced paper | +1.5 to +3.0 |
| assemble_paper | Insufficient questions | -0.5 |
| Episode ends at 50 steps without assembly | | -1.0 |

---

## TESTING CHECKLIST

Before submission, verify:
- [ ] `python -c "from models import ExamForgeAction"` works
- [ ] `python -c "from server.environment import ExamForgeEnvironment; e=ExamForgeEnvironment(); print(e.reset())"` works  
- [ ] `docker build -f server/Dockerfile -t examforge .` builds successfully
- [ ] `python inference.py` runs end-to-end without errors
- [ ] `bash validate-submission.sh` passes all checks
- [ ] README.md is complete and clearly explains the environment
- [ ] openenv.yaml has correct metadata

---

*This file was generated as part of the ExamForge submission for the Meta PyTorch OpenEnv Hackathon 2026.*
