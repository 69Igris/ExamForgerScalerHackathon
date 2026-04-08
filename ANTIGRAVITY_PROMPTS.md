# ANTIGRAVITY PROMPTS — ExamForge OpenEnv Environment
# Use Claude Opus 4 in AntiGravity for best results
# Follow EXACTLY in this order — do not skip steps

===========================================================================
STEP 0 — BEFORE YOU START (Do this manually, NOT in AntiGravity)
===========================================================================

Run in your terminal:
  pip install openenv-core
  openenv init examforge_env
  cd examforge_env

Then drag the AGENT_INSTRUCTIONS.md file INTO the examforge_env folder.
Open the examforge_env folder in AntiGravity (your IDE).
Now give AntiGravity the prompts below in sequence.

===========================================================================
PROMPT 1 — PROJECT CONTEXT SETUP (Give this FIRST in AntiGravity)
===========================================================================

Paste this entire block as your first message to AntiGravity:

---
I am building a hackathon submission called **ExamForge** for the Meta PyTorch 
OpenEnv Hackathon 2026. Please read the file `AGENT_INSTRUCTIONS.md` in this 
project carefully — it contains the complete specification for everything we need 
to build. That file is your single source of truth.

This project uses the **OpenEnv framework** by Meta + HuggingFace. Key things to know:
- Package to install: `openenv-core`
- Core imports come from: `openenv.core`
- The 3 required methods are ONLY: `reset()`, `step(action)`, `state()`
- Server runs as a FastAPI app inside Docker on port 7860
- Client connects via WebSocket using EnvClient base class
- All data models use Pydantic dataclasses inheriting from Action/Observation

Do NOT start writing any code yet. First, confirm you have understood the 
AGENT_INSTRUCTIONS.md by summarizing:
1. What ExamForge does (one sentence)
2. The 4 action types
3. The reward for a successful assemble_paper
4. The exact file structure we need to create

Only proceed once you have correctly summarized these 4 points.
---

===========================================================================
PROMPT 2 — BUILD MODELS.PY (After AntiGravity confirms understanding)
===========================================================================

---
Now let's start building. Create the file `models.py` in the root of the project.

Requirements (all from AGENT_INSTRUCTIONS.md):
- Import from `openenv.core`: Action, Observation
- Create `ActionType` enum with 4 values: generate_question, validate_question, 
  flag_question, assemble_paper
- Create `ExamForgeAction(Action)` dataclass with ALL fields listed in the 
  AGENT_INSTRUCTIONS.md — every field must be Optional with a default of None
  EXCEPT action_type which is required
- Create `ExamForgeObservation(Observation)` dataclass with ALL fields listed —
  use `field(default_factory=dict)` for dict fields and 
  `field(default_factory=list)` for list fields
- Add proper type hints for everything
- Add a docstring to each class explaining what it represents

After creating models.py, verify it works by running:
  python -c "from models import ExamForgeAction, ExamForgeObservation, ActionType; print('models.py OK')"

Show me the complete file and the verification output.
---

===========================================================================
PROMPT 3 — BUILD SERVER/ENVIRONMENT.PY (Core Logic)
===========================================================================

---
Now create `server/environment.py`. This is the most important file.

It must contain:

1. The `SUBJECT_TOPICS` dictionary at the top (copy exact content from 
   AGENT_INSTRUCTIONS.md — 5 subjects, 10 topics each)

2. A `QuestionRecord` dataclass (internal storage, NOT an OpenEnv model):
   - question_id: str
   - topic: str
   - difficulty: str  (easy/medium/hard)
   - marks: int
   - question_text: str
   - options: dict  ({"A": ..., "B": ..., "C": ..., "D": ...})
   - correct_option: str
   - explanation: str
   - is_validated: bool = False
   - is_flagged: bool = False
   - validation_score: float = 0.0

3. Class `ExamForgeEnvironment(Environment)` with:
   
   `__init__`: Initialize empty state variables
   
   `reset()`:
   - Pick a random subject from SUBJECT_TOPICS
   - Set self.current_subject, self.available_topics
   - Set paper constraints: total_marks=100, num_questions=25
   - Initialize self.question_bank = {} (empty dict, key=question_id)
   - Initialize counters: self.step_count=0, self.marks_used=0
   - Generate a UUID for self.episode_id
   - Return ExamForgeObservation with available_topics, paper_constraints filled
   
   `step(action: ExamForgeAction)`:
   Implement ALL 4 action handlers as described in AGENT_INSTRUCTIONS.md.
   Each handler returns an ExamForgeObservation.
   
   GENERATE_QUESTION handler:
   - Validate all required fields present (topic, difficulty, marks, question_text,
     option_a through option_d, correct_option, explanation)
   - Validate topic is in self.available_topics
   - Validate difficulty is "easy", "medium", or "hard"
   - Validate marks is 1, 2, or 4
   - Validate self.marks_used + marks <= 100
   - If any validation fails: return obs with reward=-0.2, last_action_success=False
   - If valid: create QuestionRecord with uuid4() as question_id
   - Add to self.question_bank
   - Update self.marks_used
   - Return obs with reward=0.3, question_id_generated set, last_action_success=True
   
   VALIDATE_QUESTION handler:
   - Check question_id exists in self.question_bank
   - If not found: return obs with reward=0.0, last_action_success=False
   - Run 3 programmatic checks (implement these as private methods):
     * _check_answer_consistency(record): explanation mentions correct option letter → +0.2
     * _check_distractor_quality(record): all 4 options have len > 3 AND are all unique → +0.3
     * _check_difficulty_calibration(record): 
         hard → explanation len > 100 chars → +0.2
         medium → explanation len > 50 chars → +0.2
         easy → any explanation → +0.2
   - Compute validation_score = sum of passed checks (0.0 to 0.7... normalize to 0-1)
   - Mark record.is_validated = True, record.validation_score = validation_score
   - Reward = validation_score * 1.0
   - Return obs with validation_score, last_action_success=True
   
   FLAG_QUESTION handler:
   - Check question_id exists
   - Check flag_reason is non-empty string with len > 10
   - If question.validation_score < 0.4 OR not yet validated: reward = +0.2
   - Else: reward = -0.1
   - Mark record.is_flagged = True
   - Return obs with last_action_success=True
   
   ASSEMBLE_PAPER handler:
   - Get valid_questions = [q for q in self.question_bank.values() if not q.is_flagged]
   - If len(valid_questions) < 10: return obs with reward=-0.5, done=False, 
     last_action_result="Need at least 10 valid questions"
   - Compute topic_coverage_score: 
       unique_topics = len(set(q.topic for q in valid_questions))
       topic_coverage_score = min(unique_topics / 5.0, 1.0)
   - Compute difficulty_distribution_score:
       counts = {easy: N, medium: N, hard: N}
       ideal = easy=0.30, medium=0.50, hard=0.20
       score = 1.0 - mean(abs(actual_ratio - ideal_ratio) for each level)
       clamp to [0, 1]
   - Compute validated_ratio = len([q for q in valid_questions if q.is_validated]) / len(valid_questions)
   - final_paper_score = 0.4*topic_coverage_score + 0.4*difficulty_distribution_score + 0.2*validated_ratio
   - reward = final_paper_score * 3.0
   - Return obs with done=True, paper_assembled=True, final_paper_score, 
     topic_coverage_score, difficulty_distribution (as dict of counts)
   
   Also handle max steps: if self.step_count >= 50, return obs with done=True, reward=-1.0
   Increment self.step_count at start of every step() call.
   
   `state()`: Return a dict with episode_id, step_count, subject, marks_used, 
   num_questions_in_bank, num_valid_questions

After writing this file, verify it works:
  python -c "
  from server.environment import ExamForgeEnvironment
  from models import ExamForgeAction
  env = ExamForgeEnvironment()
  obs = env.reset()
  print('Subject:', env.current_subject)
  print('Topics:', obs.available_topics[:3])
  print('Constraints:', obs.paper_constraints)
  print('reset() PASSED')
  "

Show me the complete file and verification output.
---

===========================================================================
PROMPT 4 — BUILD SERVER/APP.PY AND SERVER/DOCKERFILE
===========================================================================

---
Now create two files:

**File 1: `server/app.py`**

```python
from openenv.core.env_server import create_app
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ExamForgeObservation

app = create_app(
    create_environment=lambda: ExamForgeEnvironment(),
    action_class=ExamForgeAction,
    observation_class=ExamForgeObservation,
    max_concurrent_envs=32,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

If `create_app` signature is different in the installed openenv-core version, 
check with: `python -c "from openenv.core.env_server import create_app; help(create_app)"`
and adapt accordingly. The key requirement is: environment class, action class, 
observation class, and max_concurrent_envs must all be passed.

**File 2: `server/Dockerfile`**

Write a Dockerfile that:
- Uses `python:3.11-slim` as base
- Sets WORKDIR to /app
- Copies server/requirements.txt first (for Docker layer caching)
- Runs pip install
- Copies ALL project files
- Runs `pip install -e .` to install the package
- Exposes port 7860
- CMD runs uvicorn with server.app:app on 0.0.0.0:7860

**File 3: `server/requirements.txt`**
```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

Verify app.py imports work:
  python -c "from server.app import app; print('app.py OK — routes:', len(app.routes))"
---

===========================================================================
PROMPT 5 — BUILD CLIENT.PY
===========================================================================

---
Create `client.py` in the project root:

```python
from openenv.core import EnvClient
from models import ExamForgeAction, ExamForgeObservation, ActionType

class ExamForgeEnv(EnvClient):
    action_class = ExamForgeAction
    observation_class = ExamForgeObservation
```

Add these convenience methods to ExamForgeEnv:

1. `generate_question(self, topic, difficulty, marks, question_text, option_a, 
   option_b, option_c, option_d, correct_option, explanation)` → calls self.step()

2. `validate_question(self, question_id: str)` → calls self.step()

3. `flag_question(self, question_id: str, reason: str)` → calls self.step()

4. `assemble_paper(self)` → calls self.step()

Each convenience method creates the appropriate ExamForgeAction with action_type 
set correctly from the ActionType enum, then calls and returns self.step(action).

Also update `__init__.py` in the root to export:
  from .models import ExamForgeAction, ExamForgeObservation, ActionType
  from .client import ExamForgeEnv
  __all__ = ["ExamForgeAction", "ExamForgeObservation", "ActionType", "ExamForgeEnv"]

Verify:
  python -c "from examforge_env import ExamForgeAction, ExamForgeObservation, ExamForgeEnv; print('__init__.py exports OK')"
---

===========================================================================
PROMPT 6 — BUILD INFERENCE.PY (Demo Agent)
===========================================================================

---
Create `inference.py` in the project root. This is a demo agent that runs a 
complete episode WITHOUT connecting to a live server — it tests the environment 
logic directly.

Structure:
1. Import ExamForgeEnvironment from server.environment
2. Import ExamForgeAction, ActionType from models
3. Define a list called SAMPLE_QUESTIONS — hardcode exactly 15 JEE Physics MCQs.
   Each entry is a dict with keys matching ExamForgeAction fields.
   Make them realistic — actual physics questions with correct answers.
   Cover at least 5 different topics from SUBJECT_TOPICS["JEE Physics"].
   Mix of easy (5), medium (7), hard (3) difficulties.
   Marks: easy=1, medium=2, hard=4.

4. `def run_demo_episode()`:
   - Create env = ExamForgeEnvironment()
   - Force subject to "JEE Physics" for reproducibility
   - Call reset()
   - Print: "Episode started | Subject: JEE Physics"
   - Loop through SAMPLE_QUESTIONS:
     * Step 1: generate_question with this question's data
     * Print: f"Generated Q{i+1}: {topic} ({difficulty}) → reward: {obs.reward:.2f}"
     * Step 2: validate_question with the returned question_id
     * Print: f"  Validated → score: {obs.validation_score:.2f}, reward: {obs.reward:.2f}"
   - After all questions generated & validated, call assemble_paper
   - Print: f"\n=== PAPER ASSEMBLED ==="
   - Print: f"Final Paper Score: {obs.final_paper_score:.3f}"
   - Print: f"Topic Coverage: {obs.topic_coverage_score:.3f}"
   - Print: f"Difficulty Distribution: {obs.difficulty_distribution}"
   - Print: f"Total Reward from Assembly: {obs.reward:.2f}"
   - Print: f"Total Steps Used: {env.step_count}"
   - Return total accumulated reward (sum all step rewards)

5. `if __name__ == "__main__": run_demo_episode()`

Run it to verify:
  python inference.py

Show me the complete output. It should print all 15 questions being generated,
validated, and a final paper score > 0.5.
---

===========================================================================
PROMPT 7 — BUILD REMAINING FILES
===========================================================================

---
Create these final files:

**1. `openenv.yaml`** (exact content from AGENT_INSTRUCTIONS.md)

**2. `pyproject.toml`** (exact content from AGENT_INSTRUCTIONS.md)

**3. `requirements.txt`** (root level — for development):
```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

**4. `.gitignore`**:
```
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
.pytest_cache/
outputs/
*.log
```

**5. `.env`** (template only, no real secrets):
```
# ExamForge Environment Variables
# Copy this to .env and fill in values if needed
OPENENV_PORT=7860
MAX_CONCURRENT_ENVS=32
LOG_LEVEL=info
```

**6. `test_agent.py`** — Write 5 unit tests using pytest:
   - test_reset_returns_valid_observation(): calls reset(), checks available_topics is not empty
   - test_generate_valid_question(): generates 1 valid question, checks reward == 0.3
   - test_generate_invalid_question(): generates question with wrong topic, checks reward == -0.2
   - test_validate_question(): generates then validates, checks validation_score > 0
   - test_assemble_paper_insufficient(): calls assemble_paper with 0 questions, checks reward == -0.5

Run tests: python -m pytest test_agent.py -v
Show output.

**7. `validate-submission.sh`** (exact content from AGENT_INSTRUCTIONS.md)
Make it executable: chmod +x validate-submission.sh

**8. `AGENT_INSTRUCTIONS.md`** — already exists, don't touch it.

After all files are created, run: bash validate-submission.sh
Show me the complete output.
---

===========================================================================
PROMPT 8 — README.MD (Judges Read This First — Make It Perfect)
===========================================================================

---
Create a professional, impressive `README.md`. This is what Meta/HuggingFace 
judges will read first. Make it stand out.

Structure (in this exact order):

# ExamForge 🎓
> *An RL environment for training LLM agents to generate, validate, and assemble 
> high-quality competitive exam papers — targeting India's JEE, GATE, and UPSC.*

## 🎯 Why ExamForge?
- 1.5 million students appear for JEE annually
- Exam quality directly impacts students' futures
- Current AI tools hallucinate incorrect answers
- ExamForge creates a closed-loop feedback system where agents LEARN from 
  validation failures

## 🏗️ Environment Design

### Episode Lifecycle (diagram using ASCII art showing the loop):
```
reset() → generate_question → validate_question → [flag_question] → assemble_paper → done
              ↑_______________________loop__________________________________|
```

### Actions Table (markdown table with all 4 actions, their parameters, and purpose)

### Reward Structure Table (copy from AGENT_INSTRUCTIONS.md reward summary table)

### Subjects & Topics Table (list all 5 subjects with their topics)

## 🚀 Quick Start

### Installation
pip install git+https://huggingface.co/spaces/YOUR_HF_USERNAME/examforge_env

### Basic Usage (5-line code example using sync API)

### Running the Demo Agent
python inference.py

## 🧠 Training Potential
Explain how an LLM trained on this environment would:
- Learn to generate factually correct MCQs
- Learn to write high-quality distractors
- Learn to balance difficulty across a paper
- Generalize to new subjects not seen in training

## 📊 Environment Specs
| Property | Value |
|---|---|
| Max Steps per Episode | 50 |
| Target Paper Marks | 100 |
| Target Questions | 25 |
| Supported Subjects | 5 (JEE Physics, JEE Chemistry, JEE Maths, GATE CS, UPSC GS) |
| Reward Range | -1.0 to +3.0 per step |
| Evaluation | Programmatic grader + Rubric-based scoring |

## 🤝 Built With
- Meta PyTorch OpenEnv Framework
- HuggingFace Spaces (deployment)
- FastAPI + Docker (server)

## 📄 License
MIT
---

===========================================================================
PROMPT 9 — FINAL DOCKER BUILD VERIFICATION
===========================================================================

---
Let's do a final end-to-end verification. Run these commands one by one and 
show me the output of each:

1. python -m pytest test_agent.py -v

2. python inference.py

3. bash validate-submission.sh

4. docker build -f server/Dockerfile -t examforge-env:latest .
   (If Docker is not available, skip this and note it)

5. Show me the final file tree of the project:
   find . -not -path './.git/*' -not -path './__pycache__/*' -not -name '*.pyc' | sort

If any test fails, fix it before moving on. All 5 pytest tests must pass.
The inference.py must complete without errors.
The validate-submission.sh must print "ALL CHECKS PASSED".
---

===========================================================================
PROMPT 10 — HUGGINGFACE DEPLOYMENT (Final Step)
===========================================================================

---
We are ready to deploy to HuggingFace Spaces. Guide me through:

1. Login to HuggingFace CLI:
   huggingface-cli login
   (I will enter my token manually)

2. Push the environment:
   openenv push --repo-id YOUR_USERNAME/examforge_env

3. After push, verify the Space is live by fetching:
   https://huggingface.co/spaces/YOUR_USERNAME/examforge_env

If `openenv push` fails, provide the manual alternative:
- Create a new Space on HuggingFace manually
- Set SDK to "docker"
- Push the code using git

Also update the README.md to replace "YOUR_HF_USERNAME" with the actual 
HuggingFace username once deployment is done.

Final submission checklist:
- [ ] HuggingFace Space is live and accessible
- [ ] openenv.yaml is present in the Space
- [ ] README.md shows the correct install command
- [ ] inference.py output was saved as a screenshot for submission
- [ ] Submit the HuggingFace Space URL on the Scaler hackathon dashboard
---

===========================================================================
TROUBLESHOOTING — Common Issues & Fixes
===========================================================================

ISSUE 1: `ImportError: cannot import name 'Environment' from 'openenv.core'`
FIX: Check exact import path with:
     python -c "import openenv.core; print(dir(openenv.core))"
     Use whatever the actual class name is (may be BaseEnvironment or EnvBase)

ISSUE 2: `create_app` signature mismatch
FIX: Run: python -c "from openenv.core.env_server import create_app; import inspect; print(inspect.signature(create_app))"
     Adapt app.py to match exact signature

ISSUE 3: Port binding error in Docker
FIX: Ensure CMD in Dockerfile uses port 7860, not 8000

ISSUE 4: `openenv push` requires Dockerfile at root
FIX: If push fails, copy server/Dockerfile to project root and retry

ISSUE 5: Pydantic validation error on Observation fields
FIX: Make sure all fields with mutable defaults use `field(default_factory=...)`
     Never use `field: list = []` — always use `field: list = field(default_factory=list)`

ISSUE 6: `done` field conflict
FIX: The parent Observation class already has `done: bool = False`
     Do NOT redefine it in ExamForgeObservation — just set self.done = True in environment

ISSUE 7: `reward` field conflict  
FIX: Same as above — parent has `reward` field, just set it, don't redefine

===========================================================================
END OF ANTIGRAVITY PROMPTS
===========================================================================
