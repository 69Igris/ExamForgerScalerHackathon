"""
test_agent.py — ExamForge Unit Test Suite
==========================================
Tests environment logic, graders, models, and spec compliance.
Run: python -m pytest test_agent.py -v
"""

import sys
import pytest
sys.path.insert(0, ".")

from server.environment import (
    ExamForgeEnvironment,
    SUBJECT_TOPICS,
    question_generation_grader,
    question_validation_grader,
    paper_assembly_grader,
)
from models import ExamForgeAction, ExamForgeObservation, ActionType


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    e = ExamForgeEnvironment()
    e.reset()
    return e


@pytest.fixture
def physics_env():
    """Environment fixed to JEE Physics for reproducible tests."""
    e = ExamForgeEnvironment()
    e.reset()
    e.current_subject = "JEE Physics"
    e.available_topics = list(SUBJECT_TOPICS["JEE Physics"])
    e.question_bank = {}
    e.step_count = 0
    e.marks_used = 0
    e._paper_assembled = False
    return e


def make_good_question(topic="Kinematics", difficulty="medium", marks=2):
    """Create a well-formed ExamForgeAction for a valid question."""
    return ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic=topic,
        difficulty=difficulty,
        marks=marks,
        question_text="A particle undergoes uniform circular motion. What is always true?",
        option_a="Its speed is constant",
        option_b="Its velocity is constant",
        option_c="Its acceleration is zero",
        option_d="Its angular displacement is zero",
        correct_option="A",
        explanation="Option A is correct. In uniform circular motion, the speed (magnitude of velocity) "
                    "remains constant, but velocity direction changes — so velocity is NOT constant. "
                    "Centripetal acceleration is always present, so acceleration is not zero.",
    )


# ─── Test 1: reset() returns valid observation ─────────────────────────────────

def test_reset_returns_valid_observation():
    """reset() must return clean state with non-empty topics and constraints."""
    env = ExamForgeEnvironment()
    obs = env.reset()

    assert isinstance(obs, ExamForgeObservation), "reset() must return ExamForgeObservation"
    assert len(obs.available_topics) > 0, "available_topics must not be empty"
    assert len(obs.paper_constraints) > 0, "paper_constraints must not be empty"
    assert obs.questions_generated == 0, "questions_generated must be 0"
    assert obs.total_marks_used == 0, "marks_used must be 0"
    assert obs.step_count == 0, "step_count must be 0"
    assert env.question_bank == {}, "question_bank must be empty"
    assert obs.done is False, "done must be False after reset"


# ─── Test 2: Valid question generation gives +0.3 reward ───────────────────────

def test_generate_valid_question_reward(physics_env):
    """A well-formed generate_question action should yield reward == 0.3."""
    action = make_good_question("Kinematics", "medium", 2)
    obs = physics_env.step(action)

    assert obs.last_action_success is True, \
        f"Valid question should succeed. Error: {obs.last_action_result}"
    assert abs(obs.reward - 0.3) < 0.01, \
        f"Expected reward 0.3, got {obs.reward}"
    assert obs.question_id_generated is not None, \
        "question_id_generated must be set on success"
    assert obs.questions_generated == 1


# ─── Test 3: Invalid topic gives negative reward ───────────────────────────────

def test_generate_invalid_topic_penalized(physics_env):
    """Generating a question with a topic not in available_topics should penalize."""
    action = ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic="Underwater Basket Weaving",  # clearly invalid
        difficulty="easy",
        marks=1,
        question_text="This should fail",
        option_a="A", option_b="B", option_c="C", option_d="D",
        correct_option="A",
        explanation="Option A is correct for this test.",
    )
    obs = physics_env.step(action)

    assert obs.last_action_success is False, "Invalid topic must not succeed"
    assert obs.reward < 0, f"Invalid topic must give negative reward, got {obs.reward}"


# ─── Test 4: validate_question scores a valid question ────────────────────────

def test_validate_question_returns_score(physics_env):
    """After generating a good question, validate_question should return score > 0."""
    gen_action = make_good_question()
    gen_obs = physics_env.step(gen_action)
    assert gen_obs.last_action_success, "Generation must succeed before validation"
    qid = gen_obs.question_id_generated

    val_action = ExamForgeAction(
        action_type=ActionType.VALIDATE_QUESTION,
        question_id=qid,
    )
    val_obs = physics_env.step(val_action)

    assert val_obs.last_action_success is True, "Validation must succeed"
    assert val_obs.validation_score > 0.0, "Validation score must be > 0 for good question"
    assert val_obs.validation_score <= 1.0, "Validation score must be <= 1.0"
    assert val_obs.reward > 0, "Validation must give positive reward"


# ─── Test 5: assemble_paper with insufficient questions gives -0.5 ─────────────

def test_assemble_paper_insufficient_questions(physics_env):
    """Calling assemble_paper with < 10 valid questions must return reward == -0.5."""
    # Add only 3 questions — not enough to assemble
    for topic in ["Kinematics", "Optics", "Thermodynamics"]:
        action = make_good_question(topic=topic, difficulty="easy", marks=1)
        physics_env.step(action)

    assemble_action = ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER)
    obs = physics_env.step(assemble_action)

    assert abs(obs.reward - (-0.5)) < 0.01, \
        f"Expected reward -0.5 for insufficient questions, got {obs.reward}"
    assert obs.done is False, "Episode must not end on failed assembly"


# ─── Test 6: step_count increments and max_steps enforced ─────────────────────

def test_step_count_and_max_steps(physics_env):
    """step_count must increment, and done=True must trigger at step 50."""
    assert physics_env.step_count == 0

    action = ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER)
    obs = physics_env.step(action)
    assert physics_env.step_count == 1, f"step_count should be 1, got {physics_env.step_count}"

    # Force step count to 49
    physics_env.step_count = 49
    obs = physics_env.step(action)
    assert obs.done is True, "Episode must end when step_count reaches 50"


# ─── Test 7: state() returns complete dict for graders ────────────────────────

def test_state_returns_grader_compatible_dict(physics_env):
    """state() must return a dict with all keys needed by grader functions."""
    s = physics_env.state()

    assert isinstance(s, dict), "state() must return dict"
    required_keys = ["question_bank", "paper_assembled", "marks_used",
                     "step_count", "episode_id"]
    for key in required_keys:
        assert key in s, f"state() missing key: {key}"


# ─── Test 8: grader functions return valid scores ─────────────────────────────

def test_graders_return_valid_scores():
    """All 3 grader functions must return float in [0.0, 1.0] for any input."""
    # Empty state
    empty_state = {"question_bank": {}, "paper_assembled": False, "marks_used": 0}

    for grader_fn, name in [
        (question_generation_grader, "question_generation_grader"),
        (question_validation_grader, "question_validation_grader"),
        (paper_assembly_grader, "paper_assembly_grader"),
    ]:
        score = grader_fn(empty_state)
        assert isinstance(score, float), f"{name} must return float"
        assert 0.0 <= score <= 1.0, f"{name} score {score} outside [0, 1]"


# ─── Test 9: flag_question correctly rewards true positives ───────────────────

def test_flag_low_quality_question_rewarded(physics_env):
    """Flagging a question with low validation score should give +0.2 reward."""
    # Generate a question with trivial options (<=3 chars) and short explanation
    # This ensures distractor quality check FAILS → validation_score < 0.4
    action = ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic="Kinematics",
        difficulty="hard",   # hard but short explanation = low quality
        marks=4,
        question_text="What is velocity?",
        option_a="v",
        option_b="a",
        option_c="F",
        option_d="x",
        correct_option="A",
        explanation="v",   # Very short — will fail difficulty calibration AND answer consistency
    )
    gen_obs = physics_env.step(action)
    assert gen_obs.last_action_success
    qid = gen_obs.question_id_generated

    # Validate it first
    val_action = ExamForgeAction(action_type=ActionType.VALIDATE_QUESTION, question_id=qid)
    val_obs = physics_env.step(val_action)

    # If validation score is low, flagging should give +0.2
    record = physics_env.question_bank[qid]
    if record.validation_score < 0.4:
        flag_action = ExamForgeAction(
            action_type=ActionType.FLAG_QUESTION,
            question_id=qid,
            flag_reason="Explanation too short for hard difficulty question",
        )
        flag_obs = physics_env.step(flag_action)
        assert abs(flag_obs.reward - 0.2) < 0.01, \
            f"Expected +0.2 for correct flagging, got {flag_obs.reward}"
    else:
        pytest.skip("Validation score not low enough to test flagging reward")


# ─── Test 10: Full episode simulation ─────────────────────────────────────────

def test_full_episode_reaches_assembly(physics_env):
    """A complete episode: generate 10+ questions, validate, then assemble."""
    topics_plan = [
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
    ]

    generated_ids = []
    for topic, difficulty, marks in topics_plan:
        action = make_good_question(topic=topic, difficulty=difficulty, marks=marks)
        obs = physics_env.step(action)
        if obs.last_action_success and obs.question_id_generated:
            generated_ids.append(obs.question_id_generated)

    assert len(generated_ids) >= 10, \
        f"Should generate at least 10 questions, got {len(generated_ids)}"

    # Validate all
    for qid in generated_ids:
        physics_env.step(ExamForgeAction(
            action_type=ActionType.VALIDATE_QUESTION, question_id=qid
        ))

    # Assemble
    obs = physics_env.step(ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER))

    assert obs.done is True, "Episode must end after successful assembly"
    assert obs.paper_assembled is True
    assert obs.reward > 0, f"Successful assembly must give positive reward, got {obs.reward}"
    assert obs.final_paper_score > 0, "final_paper_score must be > 0"

    # Verify grader works on this state
    state = physics_env.state()
    grader_score = paper_assembly_grader(state)
    assert 0.0 <= grader_score <= 1.0, f"Grader score {grader_score} out of range"
    print(f"\n  Full episode grader score: {grader_score:.3f}")
