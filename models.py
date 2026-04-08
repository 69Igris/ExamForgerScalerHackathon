"""
ExamForge Models — Pydantic Action and Observation dataclasses for OpenEnv.

Defines the data contracts between agents and the ExamForge environment.
All models inherit from openenv.core base classes (Pydantic BaseModel).
"""

from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import Field
from openenv.core import Action, Observation


class ActionType(str, Enum):
    """The four action types available to an agent in ExamForge."""
    GENERATE_QUESTION = "generate_question"
    VALIDATE_QUESTION = "validate_question"
    FLAG_QUESTION = "flag_question"
    ASSEMBLE_PAPER = "assemble_paper"


class ExamForgeAction(Action):
    """
    An action taken by the agent in the ExamForge environment.

    Depending on action_type, different fields are required:
    - generate_question: topic, difficulty, marks, question_text, option_a-d, correct_option, explanation
    - validate_question: question_id
    - flag_question: question_id, flag_reason
    - assemble_paper: no extra fields
    """
    action_type: ActionType

    # For generate_question
    topic: Optional[str] = None
    difficulty: Optional[str] = None       # "easy" | "medium" | "hard"
    marks: Optional[int] = None            # 1, 2, or 4
    question_text: Optional[str] = None
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None
    option_d: Optional[str] = None
    correct_option: Optional[str] = None   # "A" | "B" | "C" | "D"
    explanation: Optional[str] = None

    # For validate_question / flag_question
    question_id: Optional[str] = None
    flag_reason: Optional[str] = None


class ExamForgeObservation(Observation):
    """
    The observation returned to the agent after each step in ExamForge.

    Contains episode context, paper state, action results, validation feedback,
    paper assembly metrics, and available topics/constraints.

    Note: 'done' and 'reward' are inherited from the parent Observation class.
    """
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
    last_action_result: str = ""
    last_action_success: bool = False
    question_id_generated: Optional[str] = None

    # Validation feedback
    validation_score: float = 0.0
    distractor_quality_score: float = 0.0
    difficulty_accuracy: bool = False

    # Paper assembly feedback
    paper_assembled: bool = False
    topic_coverage_score: float = 0.0
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    final_paper_score: float = 0.0

    # Topics available in this episode
    available_topics: List[str] = Field(default_factory=list)
    paper_constraints: Dict[str, Any] = Field(default_factory=dict)
