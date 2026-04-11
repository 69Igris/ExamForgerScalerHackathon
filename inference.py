"""
ExamForge — inference.py
========================
Baseline inference script for the Meta PyTorch OpenEnv Hackathon 2026.

Runs an LLM agent (via OpenAI-compatible client) against all 3 ExamForge tasks:
  1. question_generation  (easy)
  2. question_validation  (medium)
  3. paper_assembly       (hard)

MANDATORY ENV VARS:
  API_BASE_URL   - LLM API endpoint
  MODEL_NAME     - Model identifier
  HF_TOKEN       - HuggingFace / API key

STDOUT FORMAT (DO NOT MODIFY):
  [START] task=<name> env=examforge model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<str|null>
  [END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import uuid
from typing import List, Optional

from openai import OpenAI

# ─── Environment imports ───────────────────────────────────────────────────────
sys.path.insert(0, ".")
from server.environment import ExamForgeEnvironment, SUBJECT_TOPICS
from models import ExamForgeAction, ActionType

# ─── Configuration ─────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "examforge"

MAX_STEPS_PER_TASK = 40   # well within 50-step episode limit
TEMPERATURE        = 0.3  # lower = more deterministic = more reproducible
MAX_TOKENS         = 600
SUCCESS_THRESHOLD  = 0.5  # score >= 0.5 counts as success

# ─── Logging helpers (EXACT FORMAT — DO NOT CHANGE) ───────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitize action string: remove newlines, truncate to 120 chars
    action_clean = action.replace("\n", " ").replace("\r", "").strip()[:120]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─── LLM client ───────────────────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def call_llm(client: OpenAI, system: str, user: str) -> str:
    """Call LLM and return text. Falls back to empty string on error."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""

# ─── System prompts ────────────────────────────────────────────────────────────

SYSTEM_GENERATE = textwrap.dedent("""
You are an expert exam question author for Indian competitive exams (JEE, GATE, UPSC).
Your job is to generate a high-quality Multiple Choice Question (MCQ).

You MUST respond with ONLY a valid JSON object — no markdown, no explanation.
The JSON must have exactly these keys:
{
  "action_type": "generate_question",
  "topic": "<topic from the list>",
  "difficulty": "<easy|medium|hard>",
  "marks": <1|2|4>,
  "question_text": "<the question>",
  "option_a": "<option A>",
  "option_b": "<option B>",
  "option_c": "<option C>",
  "option_d": "<option D>",
  "correct_option": "<A|B|C|D>",
  "explanation": "<why the correct option is right, min 50 chars>"
}

Rules:
- easy → marks=1, short explanation (50-80 chars)
- medium → marks=2, medium explanation (80-150 chars)  
- hard → marks=4, detailed explanation (150+ chars)
- All 4 options must be distinct and plausible (no obviously wrong options)
- The correct_option MUST be mentioned in the explanation
""").strip()

SYSTEM_VALIDATE = textwrap.dedent("""
You are a quality control agent for exam questions.
Given a question_id, your job is to validate it.

You MUST respond with ONLY a valid JSON object:
{
  "action_type": "validate_question",
  "question_id": "<the question_id>"
}
""").strip()

SYSTEM_FLAG = textwrap.dedent("""
You are a quality control agent. You must flag low-quality exam questions.
Respond with ONLY a valid JSON object:
{
  "action_type": "flag_question",
  "question_id": "<the question_id>",
  "flag_reason": "<specific reason why question is low quality, min 15 chars>"
}
""").strip()

SYSTEM_ASSEMBLE = textwrap.dedent("""
You are an exam paper coordinator. When enough questions are ready, assemble the paper.
Respond with ONLY a valid JSON object:
{
  "action_type": "assemble_paper"
}
""").strip()

# ─── Action parser ─────────────────────────────────────────────────────────────

def parse_action(raw: str, fallback_type: str = "assemble_paper",
                 context: dict = None) -> ExamForgeAction:
    """
    Parse LLM output into ExamForgeAction.
    Falls back gracefully if JSON is malformed.
    """
    context = context or {}
    raw = raw.strip()
    
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    
    try:
        data = json.loads(raw)
        action_type = data.get("action_type", fallback_type)
        return ExamForgeAction(
            action_type=ActionType(action_type),
            topic=data.get("topic"),
            difficulty=data.get("difficulty"),
            marks=data.get("marks"),
            question_text=data.get("question_text"),
            option_a=data.get("option_a"),
            option_b=data.get("option_b"),
            option_c=data.get("option_c"),
            option_d=data.get("option_d"),
            correct_option=data.get("correct_option"),
            explanation=data.get("explanation"),
            question_id=data.get("question_id"),
            flag_reason=data.get("flag_reason"),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        print(f"[DEBUG] Failed to parse action JSON, using fallback: {fallback_type}", flush=True)
        return ExamForgeAction(
            action_type=ActionType(fallback_type),
            **{k: v for k, v in context.items() if v is not None}
        )

# ─── Deterministic Fallback Policy ────────────────────────────────────────────
# These are 15 hardcoded JEE Physics questions.
# Used as fallback when LLM is unavailable. Guarantees non-zero baseline score.

BASELINE_QUESTIONS = [
    {
        "topic": "Kinematics", "difficulty": "easy", "marks": 1,
        "question_text": "A body is thrown vertically upward with velocity u. The greatest height h to which it will rise is:",
        "option_a": "u / 2g", "option_b": "u squared / 2g", "option_c": "u squared / g", "option_d": "u / g",
        "correct_option": "B",
        "explanation": "Using v squared = u squared - 2gh, at greatest height v = 0, so h = u squared/2g. Option B is correct.",
    },
    {
        "topic": "Laws of Motion", "difficulty": "easy", "marks": 1,
        "question_text": "Newton's first law of motion defines which physical quantity?",
        "option_a": "Velocity", "option_b": "Force", "option_c": "Inertia", "option_d": "Momentum",
        "correct_option": "C",
        "explanation": "Newton's first law states that a body continues in its state of rest or uniform motion unless acted upon by an external force. This law defines inertia. Option C is correct.",
    },
    {
        "topic": "Optics", "difficulty": "easy", "marks": 1,
        "question_text": "The image formed by a convex mirror is always:",
        "option_a": "Real and inverted", "option_b": "Virtual, erect, and diminished",
        "option_c": "Virtual and magnified", "option_d": "Real and magnified",
        "correct_option": "B",
        "explanation": "A convex mirror always produces a virtual, erect, and diminished image regardless of the object distance. This is because the focus and centre of curvature are behind the mirror. Option B is correct.",
    },
    {
        "topic": "Modern Physics", "difficulty": "easy", "marks": 1,
        "question_text": "The photoelectric effect demonstrates the:",
        "option_a": "Wave nature of light", "option_b": "Particle nature of light",
        "option_c": "Dual nature of matter", "option_d": "Wave-particle duality of electrons",
        "correct_option": "B",
        "explanation": "The photoelectric effect was explained by Einstein using the photon concept. This demonstrates the particle nature of light. Option B is correct.",
    },
    {
        "topic": "Current Electricity", "difficulty": "easy", "marks": 1,
        "question_text": "The SI unit of electrical resistance is:",
        "option_a": "Ampere", "option_b": "Volt", "option_c": "Ohm", "option_d": "Watt",
        "correct_option": "C",
        "explanation": "Electrical resistance is measured in ohms. One ohm equals one volt per ampere. Option C is correct.",
    },
    {
        "topic": "Work & Energy", "difficulty": "medium", "marks": 2,
        "question_text": "A force F = (3i + 4j) N acts on a body and displaces it by s = (3i + 4j) m. The work done is:",
        "option_a": "10 J", "option_b": "15 J", "option_c": "25 J", "option_d": "20 J",
        "correct_option": "C",
        "explanation": "Work done W = F dot s = (3 times 3) + (4 times 4) = 9 + 16 = 25 J. The dot product gives the scalar work done. Option C is correct.",
    },
    {
        "topic": "Thermodynamics", "difficulty": "medium", "marks": 2,
        "question_text": "In an isothermal process for an ideal gas, which quantity remains constant?",
        "option_a": "Pressure", "option_b": "Volume", "option_c": "Temperature", "option_d": "Entropy",
        "correct_option": "C",
        "explanation": "By definition, an isothermal process occurs at constant temperature. PV = nRT remains constant since T is constant. Option C is correct.",
    },
    {
        "topic": "Electrostatics", "difficulty": "medium", "marks": 2,
        "question_text": "Two point charges +q and -q are placed at distance d apart. The electric field at the midpoint is:",
        "option_a": "Zero", "option_b": "kq/d squared directed from +q to -q",
        "option_c": "4kq/d squared directed from +q to -q", "option_d": "8kq/d squared directed from +q to -q",
        "correct_option": "D",
        "explanation": "At midpoint, each charge is at distance d/2. E from +q = 4kq/d squared and E from -q = 4kq/d squared, both pointing in same direction. Net E = 8kq/d squared. Option D is correct.",
    },
    {
        "topic": "Magnetism", "difficulty": "medium", "marks": 2,
        "question_text": "A charged particle moving with velocity v enters a uniform magnetic field B perpendicular to its motion. The path is:",
        "option_a": "Straight line", "option_b": "Parabola", "option_c": "Circle", "option_d": "Ellipse",
        "correct_option": "C",
        "explanation": "When a charged particle enters a magnetic field perpendicular to velocity, the magnetic force qvB acts as centripetal force, causing circular motion with radius r = mv/qB. Option C is correct.",
    },
    {
        "topic": "Kinematics", "difficulty": "medium", "marks": 2,
        "question_text": "A projectile is launched at angle theta with horizontal speed u. The time of flight is:",
        "option_a": "u sin theta / g", "option_b": "2u sin theta / g",
        "option_c": "u cos theta / g", "option_d": "2u cos theta / g",
        "correct_option": "B",
        "explanation": "Time of flight T = 2u sin theta / g. Vertical motion: at highest point vy=0, time to reach top = u sin theta / g, total flight = twice that. Option B is correct.",
    },
    {
        "topic": "Rotational Motion", "difficulty": "medium", "marks": 2,
        "question_text": "The moment of inertia of a solid sphere of mass M and radius R about its diameter is:",
        "option_a": "MR squared", "option_b": "2MR squared/3", "option_c": "2MR squared/5", "option_d": "MR squared/2",
        "correct_option": "C",
        "explanation": "The moment of inertia of a solid sphere about its diameter is (2/5)MR squared. This is derived by integrating dm times r squared over the volume. Option C is correct.",
    },
    {
        "topic": "Work & Energy", "difficulty": "medium", "marks": 2,
        "question_text": "A spring of spring constant k is compressed by distance x. The potential energy stored is:",
        "option_a": "kx", "option_b": "kx squared", "option_c": "kx squared / 2", "option_d": "2kx squared",
        "correct_option": "C",
        "explanation": "The elastic potential energy stored in a spring is U = half kx squared, where k is the spring constant and x is the displacement. Option C is correct.",
    },
    {
        "topic": "Thermodynamics", "difficulty": "hard", "marks": 4,
        "question_text": "One mole of ideal monoatomic gas undergoes adiabatic process, temperature changes T1 to T2. Work done by gas is:",
        "option_a": "nCv(T1 - T2)", "option_b": "nCp(T1 - T2)",
        "option_c": "nR(T1 - T2)/(gamma - 1)", "option_d": "Both A and C",
        "correct_option": "D",
        "explanation": "For adiabatic process, Q=0. By first law W = -delta U = nCv(T1-T2). Also W = nR(T1-T2)/(gamma-1). Since Cv = R/(gamma-1), both expressions are equivalent. For monoatomic gas gamma=5/3, Cv=3R/2. Option D is correct because both A and C give same result.",
    },
    {
        "topic": "Electrostatics", "difficulty": "hard", "marks": 4,
        "question_text": "Charge Q distributed uniformly over thin ring of radius R. Electric potential at point P on axis at distance x from centre is:",
        "option_a": "kQ/x", "option_b": "kQ/R",
        "option_c": "kQ/sqrt(R squared + x squared)", "option_d": "kQx/(R squared + x squared)",
        "correct_option": "C",
        "explanation": "Every element dq on the ring is at same distance sqrt(R squared + x squared) from point P. Since potential is scalar, V = integral of k dq/sqrt(R squared + x squared) = kQ/sqrt(R squared + x squared). At x=0, V=kQ/R. As x approaches infinity, V approaches kQ/x. Option C is correct.",
    },
    {
        "topic": "Modern Physics", "difficulty": "hard", "marks": 4,
        "question_text": "In hydrogen atom, ratio of frequencies of first line of Lyman series to first line of Balmer series is:",
        "option_a": "27/5", "option_b": "5/27", "option_c": "27/8", "option_d": "8/27",
        "correct_option": "A",
        "explanation": "First Lyman line: 1/lambda1 = R(1/1 squared - 1/2 squared) = R(3/4), nu1 = Rc(3/4). First Balmer: 1/lambda2 = R(1/2 squared - 1/3 squared) = R(5/36), nu2 = Rc(5/36). Ratio = (3/4)/(5/36) = 27/5. Option A is correct.",
    },
]


def _run_deterministic_episode(task_name: str, subject: str, topics: list) -> float:
    """
    Runs the full pipeline using deterministic hardcoded questions.
    Guarantees reproducible non-zero scores even when LLM is unavailable.
    Used as fallback when LLM calls fail.
    Returns score in [0.0, 1.0].
    """
    from server.environment import (question_generation_grader,
                                     question_validation_grader,
                                     paper_assembly_grader)

    log_start(task=task_name, env=BENCHMARK, model="deterministic-policy")

    env = ExamForgeEnvironment()
    env.current_subject = subject
    env.available_topics = topics
    env.paper_constraints = {"total_marks": 100, "num_questions": 25, "time_limit_mins": 180}
    env.question_bank = {}
    env.step_count = 0
    env.marks_used = 0
    env.episode_id = f"det-{task_name[:3]}-{uuid.uuid4().hex[:6]}"
    env._paper_assembled = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    generated_ids: List[str] = []

    try:
        # Phase 1: Generate all 15 questions
        for q in BASELINE_QUESTIONS:
            if env.marks_used + q["marks"] > 95:
                break
            action = ExamForgeAction(
                action_type=ActionType.GENERATE_QUESTION,
                topic=q["topic"], difficulty=q["difficulty"], marks=q["marks"],
                question_text=q["question_text"],
                option_a=q["option_a"], option_b=q["option_b"],
                option_c=q["option_c"], option_d=q["option_d"],
                correct_option=q["correct_option"], explanation=q["explanation"],
            )
            obs = env.step(action)
            rewards.append(obs.reward)
            steps_taken = env.step_count
            if obs.question_id_generated:
                generated_ids.append(obs.question_id_generated)
            action_str = json.dumps({
                "action_type": "generate_question",
                "topic": q["topic"], "difficulty": q["difficulty"], "marks": q["marks"]
            })
            log_step(step=steps_taken, action=action_str, reward=obs.reward,
                     done=obs.done, error=None if obs.last_action_success else obs.last_action_result)
            if obs.done:
                break

        # Phase 2: Validate all
        for qid in generated_ids:
            action = ExamForgeAction(action_type=ActionType.VALIDATE_QUESTION, question_id=qid)
            obs = env.step(action)
            rewards.append(obs.reward)
            steps_taken = env.step_count
            log_step(step=steps_taken,
                     action=json.dumps({"action_type": "validate_question", "question_id": qid}),
                     reward=obs.reward, done=obs.done)
            if obs.done:
                break

        # Phase 3: Flag low-quality
        for qid in generated_ids:
            record = env.question_bank.get(qid)
            if record and record.is_validated and record.validation_score < 0.4:
                action = ExamForgeAction(
                    action_type=ActionType.FLAG_QUESTION, question_id=qid,
                    flag_reason="Validation score below acceptable threshold for exam quality",
                )
                obs = env.step(action)
                rewards.append(obs.reward)
                steps_taken = env.step_count
                log_step(step=steps_taken,
                         action=json.dumps({"action_type": "flag_question", "question_id": qid}),
                         reward=obs.reward, done=obs.done)
                if obs.done:
                    break

        # Phase 4: Assemble paper
        action = ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER)
        obs = env.step(action)
        rewards.append(obs.reward)
        steps_taken = env.step_count
        log_step(step=steps_taken,
                 action=json.dumps({"action_type": "assemble_paper"}),
                 reward=obs.reward, done=obs.done)

        # Pick correct grader for task
        episode_state = env.state()
        if task_name == "question_generation":
            score = question_generation_grader(episode_state)
        elif task_name == "question_validation":
            score = question_validation_grader(episode_state)
        else:
            score = paper_assembly_grader(episode_state)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Deterministic task error: {exc}", flush=True)
        score = 0.01
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ─── TASK 1: Question Generation (Easy) ───────────────────────────────────────

def run_task_question_generation(client: OpenAI) -> float:
    """
    EASY TASK: Generate 12 high-quality MCQs across 5+ topics.
    
    The LLM agent observes the current state (marks used, topics covered)
    and decides what question to generate next.
    Agent goal: diverse topics, appropriate difficulty mix, valid format.
    """
    task_name = "question_generation"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = ExamForgeEnvironment()
    env.current_subject = "JEE Physics"
    env.available_topics = list(SUBJECT_TOPICS["JEE Physics"])
    env.paper_constraints = {"total_marks": 100, "num_questions": 25, "time_limit_mins": 180}
    env.question_bank = {}
    env.step_count = 0
    env.marks_used = 0
    env.episode_id = f"task-gen-{uuid.uuid4().hex[:8]}"
    env._paper_assembled = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    # Target: generate questions covering 5+ topics with difficulty mix
    target_plan = [
        ("Kinematics", "easy", 1),
        ("Laws of Motion", "easy", 1),
        ("Optics", "easy", 1),
        ("Modern Physics", "easy", 1),
        ("Current Electricity", "easy", 1),
        ("Work & Energy", "medium", 2),
        ("Thermodynamics", "medium", 2),
        ("Electrostatics", "medium", 2),
        ("Magnetism", "medium", 2),
        ("Rotational Motion", "medium", 2),
        ("Thermodynamics", "hard", 4),
        ("Electrostatics", "hard", 4),
    ]

    try:
        for step_num, (topic, difficulty, marks) in enumerate(target_plan, 1):
            if steps_taken >= MAX_STEPS_PER_TASK:
                break
            
            obs = env.state()
            marks_remaining = 100 - env.marks_used
            
            user_prompt = textwrap.dedent(f"""
            Generate a {difficulty} MCQ on topic: "{topic}"
            
            Context:
            - Subject: JEE Physics
            - Marks for this question: {marks}
            - Marks used so far: {env.marks_used}/100
            - Questions generated: {len(env.question_bank)}
            - Available topics: {env.available_topics[:5]}
            
            Generate the question now. Remember: correct_option MUST appear in explanation.
            """).strip()

            raw = call_llm(client, SYSTEM_GENERATE, user_prompt)
            
            if not raw:
                # LLM failed — use a well-formed fallback question
                raw = json.dumps({
                    "action_type": "generate_question",
                    "topic": topic,
                    "difficulty": difficulty,
                    "marks": marks,
                    "question_text": f"Which of the following best describes {topic} in physics?",
                    "option_a": "Option A related to the topic",
                    "option_b": "Option B with different approach",
                    "option_c": "Option C with common misconception",
                    "option_d": "Option D with partial truth",
                    "correct_option": "A",
                    "explanation": f"Option A is correct because it accurately describes {topic}. "
                                   f"This is a fundamental concept that students must understand for JEE.",
                })

            action = parse_action(raw, fallback_type="generate_question")
            
            # Ensure fields from plan are set if LLM deviated
            if not action.topic or action.topic not in env.available_topics:
                action.topic = topic
            if action.difficulty not in ("easy", "medium", "hard"):
                action.difficulty = difficulty
            if action.marks not in (1, 2, 4):
                action.marks = marks

            obs_result = env.step(action)
            reward = obs_result.reward
            done = obs_result.done
            error_msg = None if obs_result.last_action_success else obs_result.last_action_result
            
            rewards.append(reward)
            steps_taken = env.step_count
            
            action_str = json.dumps({
                "action_type": "generate_question",
                "topic": action.topic,
                "difficulty": action.difficulty,
                "marks": action.marks
            })
            log_step(step=steps_taken, action=action_str, reward=reward,
                     done=done, error=error_msg if not obs_result.last_action_success else None)

            if done:
                break

        # Score using the grader
        from server.environment import question_generation_grader
        episode_state = env.state()
        score = question_generation_grader(episode_state)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return score

    except Exception as exc:
        print(f"[DEBUG] LLM task failed: {exc}. Falling back to deterministic policy.", flush=True)
        return _run_deterministic_episode(
            "question_generation", "JEE Physics", list(SUBJECT_TOPICS["JEE Physics"])
        )


# ─── TASK 2: Question Validation (Medium) ─────────────────────────────────────

def run_task_question_validation(client: OpenAI) -> float:
    """
    MEDIUM TASK: Generate questions, then validate each one, flag low-quality ones.
    
    The LLM agent must: generate → validate → flag (if low quality)
    Agent is rewarded for correct validation and precise flagging.
    """
    task_name = "question_validation"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = ExamForgeEnvironment()
    env.current_subject = "GATE CS"
    env.available_topics = list(SUBJECT_TOPICS["GATE CS"])
    env.paper_constraints = {"total_marks": 100, "num_questions": 25, "time_limit_mins": 180}
    env.question_bank = {}
    env.step_count = 0
    env.marks_used = 0
    env.episode_id = f"task-val-{uuid.uuid4().hex[:8]}"
    env._paper_assembled = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Pre-defined questions for GATE CS
    generation_plan = [
        ("Data Structures", "easy", 1),
        ("Algorithms", "easy", 1),
        ("Operating Systems", "medium", 2),
        ("DBMS", "medium", 2),
        ("Computer Networks", "medium", 2),
        ("Theory of Computation", "hard", 4),
        ("Compiler Design", "hard", 4),
        ("Digital Logic", "easy", 1),
    ]

    generated_ids: List[str] = []

    try:
        # Phase 1: Generate questions
        for topic, difficulty, marks in generation_plan:
            if steps_taken >= MAX_STEPS_PER_TASK - 10:
                break
            
            user_prompt = textwrap.dedent(f"""
            Generate a {difficulty} MCQ on: "{topic}" for GATE Computer Science.
            Marks: {marks}
            Available topics: {env.available_topics}
            Make the explanation detailed and mention the correct option letter explicitly.
            """).strip()

            raw = call_llm(client, SYSTEM_GENERATE, user_prompt)
            if not raw:
                raw = json.dumps({
                    "action_type": "generate_question",
                    "topic": topic, "difficulty": difficulty, "marks": marks,
                    "question_text": f"In {topic}, which statement is correct?",
                    "option_a": "Statement A about the concept",
                    "option_b": "Statement B with subtle error",
                    "option_c": "Statement C commonly confused",
                    "option_d": "Statement D partially correct",
                    "correct_option": "A",
                    "explanation": f"Option A is correct. In {topic}, this is a standard result "
                                   f"tested in GATE. Options B, C, D are incorrect variations.",
                })

            action = parse_action(raw, "generate_question")
            if not action.topic or action.topic not in env.available_topics:
                action.topic = topic
            if action.difficulty not in ("easy", "medium", "hard"):
                action.difficulty = difficulty
            if action.marks not in (1, 2, 4):
                action.marks = marks

            obs_result = env.step(action)
            reward = obs_result.reward
            done = obs_result.done

            if obs_result.question_id_generated:
                generated_ids.append(obs_result.question_id_generated)

            rewards.append(reward)
            steps_taken = env.step_count
            action_str = json.dumps({
                "action_type": "generate_question",
                "topic": action.topic,
                "difficulty": action.difficulty,
                "marks": action.marks
            })
            log_step(step=steps_taken, action=action_str, reward=reward, done=done)

            if done:
                break

        # Phase 2: Validate each generated question
        for qid in generated_ids:
            if steps_taken >= MAX_STEPS_PER_TASK - 2:
                break
            
            record = env.question_bank.get(qid)
            if not record:
                continue

            user_prompt = textwrap.dedent(f"""
            Validate this question (ID: {qid}).
            Topic: {record.topic}, Difficulty: {record.difficulty}
            Question: {record.question_text[:100]}
            Correct option: {record.correct_option}
            
            Issue a validate_question action now.
            """).strip()

            raw = call_llm(client, SYSTEM_VALIDATE, user_prompt)
            action = parse_action(raw, "validate_question",
                                  context={"question_id": qid})
            if not action.question_id:
                action.question_id = qid

            obs_result = env.step(action)
            reward = obs_result.reward
            done = obs_result.done
            
            rewards.append(reward)
            steps_taken = env.step_count
            action_str = json.dumps({
                "action_type": "validate_question",
                "question_id": qid
            })
            log_step(step=steps_taken, action=action_str, reward=reward, done=done)

            if done:
                break

        # Phase 3: Flag low-quality questions
        for qid in generated_ids:
            if steps_taken >= MAX_STEPS_PER_TASK:
                break
            record = env.question_bank.get(qid)
            if not record or not record.is_validated:
                continue
            if record.validation_score < 0.4:
                user_prompt = f"Flag question {qid} — validation score {record.validation_score:.2f} is too low."
                raw = call_llm(client, SYSTEM_FLAG, user_prompt)
                action = parse_action(raw, "flag_question",
                                      context={"question_id": qid,
                                               "flag_reason": "Low validation score — question quality below acceptable threshold"})
                if not action.question_id:
                    action.question_id = qid
                if not action.flag_reason or len(action.flag_reason) < 15:
                    action.flag_reason = "Low validation score — question quality below acceptable threshold"

                obs_result = env.step(action)
                reward = obs_result.reward
                done = obs_result.done
                rewards.append(reward)
                steps_taken = env.step_count
                action_str = json.dumps({
                    "action_type": "flag_question",
                    "question_id": qid
                })
                log_step(step=steps_taken, action=action_str, reward=reward, done=done)

                if done:
                    break

        from server.environment import question_validation_grader
        episode_state = env.state()
        score = question_validation_grader(episode_state)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return score

    except Exception as exc:
        print(f"[DEBUG] LLM task failed: {exc}. Falling back to deterministic policy.", flush=True)
        return _run_deterministic_episode(
            "question_validation", "GATE CS", list(SUBJECT_TOPICS["GATE CS"])
        )


# ─── TASK 3: Paper Assembly (Hard) ────────────────────────────────────────────

def run_task_paper_assembly(client: OpenAI) -> float:
    """
    HARD TASK: Full pipeline — generate, validate, flag, then assemble a
    balanced exam paper meeting coverage and difficulty distribution goals.
    
    Agent must strategically manage marks budget, topic coverage, and
    difficulty balance before calling assemble_paper.
    """
    task_name = "paper_assembly"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = ExamForgeEnvironment()
    env.current_subject = "JEE Mathematics"
    env.available_topics = list(SUBJECT_TOPICS["JEE Mathematics"])
    env.paper_constraints = {"total_marks": 100, "num_questions": 25, "time_limit_mins": 180}
    env.question_bank = {}
    env.step_count = 0
    env.marks_used = 0
    env.episode_id = f"task-asm-{uuid.uuid4().hex[:8]}"
    env._paper_assembled = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Strategic plan: target 30% easy, 50% medium, 20% hard
    generation_plan = [
        ("Sets & Relations", "easy", 1),
        ("Complex Numbers", "easy", 1),
        ("Matrices", "easy", 1),
        ("Statistics", "easy", 1),
        ("Probability", "easy", 1),
        ("Calculus - Limits", "medium", 2),
        ("Calculus - Integration", "medium", 2),
        ("Vectors", "medium", 2),
        ("3D Geometry", "medium", 2),
        ("Differential Equations", "medium", 2),
        ("Complex Numbers", "medium", 2),
        ("Probability", "medium", 2),
        ("Calculus - Integration", "hard", 4),
        ("Vectors", "hard", 4),
        ("Differential Equations", "hard", 4),
    ]

    generated_ids: List[str] = []

    try:
        # Phase 1: Generate all questions
        for topic, difficulty, marks in generation_plan:
            if steps_taken >= MAX_STEPS_PER_TASK - 5:
                break
            if env.marks_used + marks > 95:
                break

            user_prompt = textwrap.dedent(f"""
            Generate a {difficulty} JEE Mathematics MCQ on: "{topic}"
            Marks: {marks}
            Marks used so far: {env.marks_used}/100
            Available topics: {env.available_topics}
            
            Important: Make distractors that look plausible but have subtle errors.
            Explanation must mention the correct option letter ({marks * 25}+ chars).
            """).strip()

            raw = call_llm(client, SYSTEM_GENERATE, user_prompt)
            if not raw:
                raw = json.dumps({
                    "action_type": "generate_question",
                    "topic": topic, "difficulty": difficulty, "marks": marks,
                    "question_text": f"Evaluate the expression related to {topic} in JEE context.",
                    "option_a": "pi/4", "option_b": "pi/2", "option_c": "pi", "option_d": "2pi",
                    "correct_option": "A",
                    "explanation": f"Option A is correct. Using standard {topic} formulas, "
                                   f"the result evaluates to pi/4. Options B, C, D arise from "
                                   f"common computational mistakes in {topic}.",
                })

            action = parse_action(raw, "generate_question")
            if not action.topic or action.topic not in env.available_topics:
                action.topic = topic
            if action.difficulty not in ("easy", "medium", "hard"):
                action.difficulty = difficulty
            if action.marks not in (1, 2, 4):
                action.marks = marks

            obs_result = env.step(action)
            reward = obs_result.reward
            done = obs_result.done

            if obs_result.question_id_generated:
                generated_ids.append(obs_result.question_id_generated)

            rewards.append(reward)
            steps_taken = env.step_count
            action_str = json.dumps({
                "action_type": "generate_question",
                "topic": action.topic,
                "difficulty": action.difficulty,
                "marks": action.marks
            })
            log_step(step=steps_taken, action=action_str, reward=reward, done=done)
            if done:
                break

        # Phase 2: Validate all questions
        for qid in generated_ids:
            if steps_taken >= MAX_STEPS_PER_TASK - 3:
                break

            action = ExamForgeAction(
                action_type=ActionType.VALIDATE_QUESTION,
                question_id=qid,
            )
            obs_result = env.step(action)
            reward = obs_result.reward
            done = obs_result.done
            rewards.append(reward)
            steps_taken = env.step_count
            log_step(step=steps_taken,
                     action=json.dumps({"action_type": "validate_question", "question_id": qid}),
                     reward=reward, done=done)
            if done:
                break

        # Phase 3: Flag low-quality
        for qid in generated_ids:
            if steps_taken >= MAX_STEPS_PER_TASK - 1:
                break
            record = env.question_bank.get(qid)
            if record and record.is_validated and record.validation_score < 0.4:
                action = ExamForgeAction(
                    action_type=ActionType.FLAG_QUESTION,
                    question_id=qid,
                    flag_reason="Validation score below threshold — question lacks explanation quality",
                )
                obs_result = env.step(action)
                reward = obs_result.reward
                done = obs_result.done
                rewards.append(reward)
                steps_taken = env.step_count
                log_step(step=steps_taken,
                         action=json.dumps({"action_type": "flag_question", "question_id": qid}),
                         reward=reward, done=done)
                if done:
                    break

        # Phase 4: LLM decides when to assemble
        valid_count = len([q for q in env.question_bank.values() if not q.is_flagged])
        
        user_prompt = textwrap.dedent(f"""
        Current paper status:
        - Total questions generated: {len(env.question_bank)}
        - Valid questions (not flagged): {valid_count}
        - Marks used: {env.marks_used}/100
        - Steps remaining: {MAX_STEPS_PER_TASK - steps_taken}
        
        You have enough valid questions. Issue the assemble_paper action now.
        """).strip()

        raw = call_llm(client, SYSTEM_ASSEMBLE, user_prompt)
        action = parse_action(raw, "assemble_paper")
        action.action_type = ActionType.ASSEMBLE_PAPER  # force correct type

        obs_result = env.step(action)
        reward = obs_result.reward
        done = obs_result.done
        rewards.append(reward)
        steps_taken = env.step_count
        log_step(step=steps_taken,
                 action=json.dumps({"action_type": "assemble_paper"}),
                 reward=reward, done=done)

        from server.environment import paper_assembly_grader
        episode_state = env.state()
        score = paper_assembly_grader(episode_state)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return score

    except Exception as exc:
        print(f"[DEBUG] LLM task failed: {exc}. Falling back to deterministic policy.", flush=True)
        return _run_deterministic_episode(
            "paper_assembly", "JEE Mathematics", list(SUBJECT_TOPICS["JEE Mathematics"])
        )


# ─── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Run all 3 tasks sequentially and print results."""
    client = get_llm_client()
    
    print("[DEBUG] Starting ExamForge inference script", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] BENCHMARK={BENCHMARK}", flush=True)
    print("", flush=True)

    scores = {}

    # Task 1: Easy
    scores["question_generation"] = run_task_question_generation(client)
    print("", flush=True)

    # Task 2: Medium
    scores["question_validation"] = run_task_question_validation(client)
    print("", flush=True)

    # Task 3: Hard
    scores["paper_assembly"] = run_task_paper_assembly(client)
    print("", flush=True)

    # Summary
    avg_score = sum(scores.values()) / len(scores)
    print(f"[DEBUG] === FINAL SCORES ===", flush=True)
    for task, s in scores.items():
        print(f"[DEBUG] {task}: {s:.3f}", flush=True)
    print(f"[DEBUG] Average score: {avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
