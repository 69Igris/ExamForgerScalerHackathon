"""
ExamForge Unit Tests — test_agent.py

5 unit tests to verify the ExamForge environment logic.
"""

import sys
sys.path.insert(0, ".")
import pytest
from server.environment import ExamForgeEnvironment, SUBJECT_TOPICS
from models import ExamForgeAction, ExamForgeObservation, ActionType


@pytest.fixture
def env():
    """Create a fresh ExamForgeEnvironment for each test."""
    e = ExamForgeEnvironment()
    e.reset(seed=42)
    return e


def test_reset_returns_valid_observation(env):
    """reset() should return an observation with non-empty available_topics."""
    obs = env.reset()
    assert isinstance(obs, ExamForgeObservation)
    assert len(obs.available_topics) > 0
    assert obs.done is False
    assert env.current_subject in SUBJECT_TOPICS
    assert obs.paper_constraints["total_marks"] == 100


def test_generate_valid_question(env):
    """Generating a well-formed question should yield reward == +0.3."""
    topic = env.available_topics[0]
    action = ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic=topic,
        difficulty="medium",
        marks=2,
        question_text="What is the SI unit of force?",
        option_a="Joule",
        option_b="Newton",
        option_c="Watt",
        option_d="Pascal",
        correct_option="B",
        explanation="The SI unit of force is the Newton, named after Sir Isaac Newton. Option B is correct.",
    )
    obs = env.step(action)
    assert obs.reward == 0.3
    assert obs.last_action_success is True
    assert obs.question_id_generated is not None


def test_generate_invalid_question(env):
    """Generating a question with an invalid topic should yield reward == -0.2."""
    action = ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic="Underwater Basket Weaving",  # invalid topic
        difficulty="easy",
        marks=1,
        question_text="What weave pattern is strongest?",
        option_a="Plain weave",
        option_b="Twill weave",
        option_c="Satin weave",
        option_d="Basket weave",
        correct_option="A",
        explanation="Plain weave is the strongest basic weave pattern.",
    )
    obs = env.step(action)
    assert obs.reward == -0.2
    assert obs.last_action_success is False


def test_validate_question(env):
    """Generating then validating a question should yield validation_score > 0."""
    topic = env.available_topics[0]
    gen_action = ExamForgeAction(
        action_type=ActionType.GENERATE_QUESTION,
        topic=topic,
        difficulty="easy",
        marks=1,
        question_text="What is 2 + 2?",
        option_a="Three",
        option_b="Four",
        option_c="Five",
        option_d="Six",
        correct_option="B",
        explanation="2 + 2 = 4, basic arithmetic. Option B is correct.",
    )
    gen_obs = env.step(gen_action)
    assert gen_obs.question_id_generated is not None

    val_action = ExamForgeAction(
        action_type=ActionType.VALIDATE_QUESTION,
        question_id=gen_obs.question_id_generated,
    )
    val_obs = env.step(val_action)
    assert val_obs.validation_score > 0
    assert val_obs.last_action_success is True


def test_assemble_paper_insufficient(env):
    """Assembling a paper with 0 questions should yield reward == -0.5."""
    action = ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER)
    obs = env.step(action)
    assert obs.reward == -0.5
    assert obs.done is False
    assert obs.paper_assembled is False
