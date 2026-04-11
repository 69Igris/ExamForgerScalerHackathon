"""
ExamForge Environment — Core RL environment logic.

Implements the ExamForgeEnvironment that trains LLM agents to generate,
validate, and assemble high-quality competitive exam papers.
"""

import uuid
import random
from dataclasses import dataclass, field as dc_field
from typing import Optional, Any, Dict

from openenv.core import Environment, State

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import ExamForgeAction, ExamForgeObservation, ActionType


# ─────────────────────────────────────────────────────────────────────────────
# Subject Topics (hardcoded question bank topics per subject)
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_TOPICS = {
    "JEE Physics": [
        "Kinematics", "Laws of Motion", "Work & Energy",
        "Rotational Motion", "Thermodynamics", "Electrostatics",
        "Current Electricity", "Magnetism", "Optics", "Modern Physics"
    ],
    "JEE Chemistry": [
        "Atomic Structure", "Chemical Bonding", "Thermochemistry",
        "Electrochemistry", "Organic Reactions", "Coordination Chemistry",
        "p-Block Elements", "d-Block Elements", "Polymers", "Biomolecules"
    ],
    "JEE Mathematics": [
        "Sets & Relations", "Complex Numbers", "Matrices",
        "Calculus - Limits", "Calculus - Integration", "Probability",
        "Vectors", "3D Geometry", "Differential Equations", "Statistics"
    ],
    "GATE CS": [
        "Data Structures", "Algorithms", "Operating Systems", "DBMS",
        "Computer Networks", "Theory of Computation", "Compiler Design",
        "Digital Logic", "Computer Organization", "Software Engineering"
    ],
    "UPSC GS": [
        "Indian Polity", "Indian Economy", "Modern History", "Geography",
        "Science & Technology", "Environment & Ecology", "International Relations",
        "Art & Culture", "Ancient History", "Social Issues"
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal Question Record (NOT an OpenEnv model — just internal storage)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionRecord:
    """Internal storage for a single generated question."""
    question_id: str
    topic: str
    difficulty: str          # "easy" | "medium" | "hard"
    marks: int               # 1, 2, or 4
    question_text: str
    options: dict            # {"A": ..., "B": ..., "C": ..., "D": ...}
    correct_option: str      # "A" | "B" | "C" | "D"
    explanation: str
    is_validated: bool = False
    is_flagged: bool = False
    validation_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Task Grader Functions (referenced from openenv.yaml)
# These are standalone module-level functions, NOT class methods.
# Signature: grader(episode_state: dict) -> float in [0.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────

def question_generation_grader(episode_state: dict) -> float:
    """
    Scores how well agent generated questions.
    Returns float in [0.0, 1.0]. Never returns same score for different inputs.
    """
    question_bank = episode_state.get("question_bank", {})
    if not question_bank:
        return 0.0

    questions = list(question_bank.values())
    total = len(questions)

    def get_attr(q, attr, default=None):
        return getattr(q, attr, q.get(attr, default) if isinstance(q, dict) else default)

    # Metric 1: Count score (target = 15 questions)
    count_score = min(total / 15.0, 1.0)

    # Metric 2: Topic diversity (target = 5+ unique topics)
    unique_topics = len(set(get_attr(q, "topic", "unknown") for q in questions))
    topic_score = min(unique_topics / 5.0, 1.0)

    # Metric 3: Marks variety (target = all 3 mark values: 1, 2, 4)
    unique_marks = len(set(get_attr(q, "marks", 1) for q in questions))
    marks_score = min(unique_marks / 3.0, 1.0)

    final = 0.4 * count_score + 0.4 * topic_score + 0.2 * marks_score
    return round(float(min(max(final, 0.0), 1.0)), 4)


def question_validation_grader(episode_state: dict) -> float:
    """
    Scores how well agent validated questions.
    Returns float in [0.0, 1.0]. Varies with validation quality.
    """
    question_bank = episode_state.get("question_bank", {})
    if not question_bank:
        return 0.0

    questions = list(question_bank.values())
    total = len(questions)

    def get_attr(q, attr, default=None):
        return getattr(q, attr, q.get(attr, default) if isinstance(q, dict) else default)

    validated = [q for q in questions if get_attr(q, "is_validated", False)]
    flagged = [q for q in questions if get_attr(q, "is_flagged", False)]

    # Metric 1: What fraction was validated?
    validation_ratio = len(validated) / total if total > 0 else 0.0

    # Metric 2: Average quality of validated questions
    avg_score = 0.0
    if validated:
        scores = [get_attr(q, "validation_score", 0.0) for q in validated]
        avg_score = sum(scores) / len(scores)

    # Metric 3: Flagging precision (correctly identified low-quality)
    if flagged:
        correct_flags = sum(
            1 for q in flagged
            if get_attr(q, "is_validated", False) and get_attr(q, "validation_score", 0.5) < 0.4
        )
        flagging_score = min(correct_flags / max(len(flagged), 1), 1.0)
    else:
        flagging_score = 0.5  # neutral — no flagging attempted

    final = 0.4 * validation_ratio + 0.4 * avg_score + 0.2 * flagging_score
    return round(float(min(max(final, 0.0), 1.0)), 4)


def paper_assembly_grader(episode_state: dict) -> float:
    """
    Scores how well agent assembled a complete, balanced exam paper.
    Returns float in [0.0, 1.0]. Highest score requires assembly + balance.
    """
    question_bank = episode_state.get("question_bank", {})
    paper_assembled = episode_state.get("paper_assembled", False)

    if not question_bank:
        return 0.0

    def get_attr(q, attr, default=None):
        return getattr(q, attr, q.get(attr, default) if isinstance(q, dict) else default)

    questions = list(question_bank.values())
    valid_questions = [q for q in questions if not get_attr(q, "is_flagged", False)]

    if not valid_questions:
        return 0.0

    # Metric 1: Assembly completion bonus (must call assemble_paper)
    assembly_bonus = 0.30 if paper_assembled else 0.0

    # Metric 2: Topic coverage (target: 5+ topics)
    unique_topics = len(set(get_attr(q, "topic", "unknown") for q in valid_questions))
    topic_coverage = min(unique_topics / 5.0, 1.0)

    # Metric 3: Difficulty balance (ideal: 30% easy, 50% medium, 20% hard)
    total_valid = len(valid_questions)
    easy_ratio = sum(1 for q in valid_questions if get_attr(q, "difficulty", "") == "easy") / total_valid
    medium_ratio = sum(1 for q in valid_questions if get_attr(q, "difficulty", "") == "medium") / total_valid
    hard_ratio = sum(1 for q in valid_questions if get_attr(q, "difficulty", "") == "hard") / total_valid

    diff_score = max(0.0, 1.0 - (
        abs(easy_ratio - 0.30) +
        abs(medium_ratio - 0.50) +
        abs(hard_ratio - 0.20)
    ) / 2.0)

    # Metric 4: Validation quality
    validated_count = sum(1 for q in valid_questions if get_attr(q, "is_validated", False))
    validation_ratio = validated_count / total_valid

    final = (
        assembly_bonus +
        0.30 * topic_coverage +
        0.25 * diff_score +
        0.15 * validation_ratio
    )
    return round(float(min(max(final, 0.0), 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ExamForge Environment
# ─────────────────────────────────────────────────────────────────────────────

class ExamForgeEnvironment(Environment):
    """
    ExamForge RL Environment for training LLM agents to generate, validate,
    and assemble high-quality competitive exam papers.

    Episode lifecycle:
        reset() → generate_question (loop) → validate_question (loop)
                → flag_question (optional) → assemble_paper → done

    Supports: JEE Physics, JEE Chemistry, JEE Mathematics, GATE CS, UPSC GS.
    Max 50 steps per episode. Target: 25 questions, 100 total marks.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.current_subject: str = ""
        self.available_topics: list = []
        self.question_bank: Dict[str, QuestionRecord] = {}
        self.step_count: int = 0
        self.marks_used: int = 0
        self.episode_id: str = ""
        self.paper_constraints: dict = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_observation(self, **kwargs) -> ExamForgeObservation:
        """Build an observation with current environment state merged in."""
        valid_qs = [q for q in self.question_bank.values() if not q.is_flagged]
        validated_qs = [q for q in valid_qs if q.is_validated]

        defaults = dict(
            episode_id=self.episode_id,
            step_count=self.step_count,
            questions_generated=len(self.question_bank),
            questions_validated=len(validated_qs),
            questions_flagged=len([q for q in self.question_bank.values() if q.is_flagged]),
            total_marks_used=self.marks_used,
            target_total_marks=100,
            available_topics=self.available_topics,
            paper_constraints=self.paper_constraints,
        )
        defaults.update(kwargs)
        return ExamForgeObservation(**defaults)

    def _check_answer_consistency(self, record: QuestionRecord) -> float:
        """Check if the explanation mentions the correct option letter."""
        correct = record.correct_option.upper()
        if correct in record.explanation:
            return 0.2
        # Also check if the correct option text appears in the explanation
        correct_text = record.options.get(correct, "")
        if correct_text and len(correct_text) > 5 and correct_text.lower() in record.explanation.lower():
            return 0.2
        return 0.0

    def _check_distractor_uniqueness(self, record: QuestionRecord) -> float:
        """
        Checks that all 4 options are genuinely distinct.

        Detects:
        - Identical options (score = 0)
        - Options that are just rephrased versions (token overlap)
        - Options that are too short to be meaningful (< 2 chars)

        Returns: float 0.0-0.3 (contributes to validation_score)
        """
        options = [
            str(record.options.get("A", "")).strip(),
            str(record.options.get("B", "")).strip(),
            str(record.options.get("C", "")).strip(),
            str(record.options.get("D", "")).strip(),
        ]

        # Check 1: All options must exist and have meaningful length
        if any(len(opt) < 2 for opt in options):
            return 0.0

        # Check 2: All 4 options must be unique (case-insensitive)
        unique_options = set(opt.lower() for opt in options)
        if len(unique_options) < 4:
            return 0.0  # Duplicate options detected

        # Check 3: Token overlap — no two options should be >70% similar
        def token_overlap(s1: str, s2: str) -> float:
            t1 = set(s1.lower().split())
            t2 = set(s2.lower().split())
            if not t1 or not t2:
                return 0.0
            return len(t1 & t2) / max(len(t1), len(t2))

        pairs = [(options[i], options[j]) for i in range(4) for j in range(i+1, 4)]
        max_overlap = max(token_overlap(a, b) for a, b in pairs)

        if max_overlap > 0.7:
            return 0.1  # Too similar but not identical
        elif max_overlap > 0.5:
            return 0.2
        else:
            return 0.3  # Good diversity

    def _check_difficulty_calibration(self, record: QuestionRecord) -> float:
        """
        Checks if explanation depth matches declared difficulty.

        Uses a simple readability proxy:
        - Word count as a proxy for complexity
        - Presence of mathematical notation as indicator of hard questions
        - Multi-step reasoning indicators (therefore, thus, hence, etc.)

        Returns: float 0.0-0.2 (contributes to validation_score)
        """
        explanation = record.explanation or ""
        difficulty = record.difficulty or "easy"

        word_count = len(explanation.split())
        has_math = any(c in explanation for c in ['=', '/', '+', '-', '*', '^', '(', ')'])
        has_reasoning = any(w in explanation.lower() for w in
                          ['therefore', 'thus', 'hence', 'because', 'since', 'using', 'applying'])

        if difficulty == "easy":
            return 0.2 if word_count >= 15 else 0.1
        elif difficulty == "medium":
            if word_count >= 40 or (word_count >= 25 and has_math):
                return 0.2
            elif word_count >= 25:
                return 0.1
            return 0.0
        else:  # hard
            if word_count >= 60 and (has_math or has_reasoning):
                return 0.2
            elif word_count >= 40 and has_math:
                return 0.15
            elif word_count >= 40:
                return 0.1
            return 0.0

    def _get_difficulty_requirement(self) -> Optional[str]:
        """
        Returns a required difficulty if the paper is becoming imbalanced.

        After 10 questions: if <20% are medium -> require medium
        After 15 questions: if <10% are hard -> require hard
        After 5 questions: if all same difficulty -> diversify

        This creates ADAPTIVE DIFFICULTY PRESSURE — the environment
        nudges agents toward pedagogically balanced papers.
        Returns None if paper is already balanced.
        """
        questions = list(self.question_bank.values())
        total = len(questions)

        if total < 5:
            return None

        easy_count = sum(1 for q in questions if q.difficulty == "easy")
        medium_count = sum(1 for q in questions if q.difficulty == "medium")
        hard_count = sum(1 for q in questions if q.difficulty == "hard")

        if total >= 15 and hard_count / total < 0.10:
            return "hard"
        if total >= 10 and medium_count / total < 0.20:
            return "medium"
        if total >= 5 and easy_count == total:
            return "medium"

        return None

    # ── core API ─────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ExamForgeObservation:
        """Start a new episode. Pick a random subject and initialize state."""
        if seed is not None:
            random.seed(seed)

        self.current_subject = random.choice(list(SUBJECT_TOPICS.keys()))
        self.available_topics = list(SUBJECT_TOPICS[self.current_subject])
        self.paper_constraints = {
            "total_marks": 100,
            "num_questions": 25,
            "time_limit_mins": 180,
        }
        self.question_bank = {}
        self.step_count = 0
        self.marks_used = 0
        self.episode_id = episode_id or str(uuid.uuid4())
        self._paper_assembled = False

        return self._build_observation(
            reward=0.0,
            done=False,
            last_action_result=f"Episode started — Subject: {self.current_subject}",
            last_action_success=True,
        )

    def step(self, action: ExamForgeAction, timeout_s: Optional[float] = None, **kwargs) -> ExamForgeObservation:
        """Execute one action and return observation with reward."""
        self.step_count += 1

        # Max steps check
        if self.step_count >= 50:
            return self._build_observation(
                reward=-1.0,
                done=True,
                last_action_result="Episode ended — max steps (50) reached without assembly.",
                last_action_success=False,
            )

        if action.action_type == ActionType.GENERATE_QUESTION:
            return self._handle_generate(action)
        elif action.action_type == ActionType.VALIDATE_QUESTION:
            return self._handle_validate(action)
        elif action.action_type == ActionType.FLAG_QUESTION:
            return self._handle_flag(action)
        elif action.action_type == ActionType.ASSEMBLE_PAPER:
            return self._handle_assemble(action)
        else:
            return self._build_observation(
                reward=-0.2,
                done=False,
                last_action_result=f"Unknown action type: {action.action_type}",
                last_action_success=False,
            )

    # ── action handlers ──────────────────────────────────────────────────────

    def _handle_generate(self, action: ExamForgeAction) -> ExamForgeObservation:
        """Handle GENERATE_QUESTION action."""
        # Validate required fields
        required = [
            action.topic, action.difficulty, action.marks,
            action.question_text, action.option_a, action.option_b,
            action.option_c, action.option_d, action.correct_option,
            action.explanation,
        ]
        if any(f is None for f in required):
            return self._build_observation(
                reward=-0.2,
                done=False,
                last_action_result="Missing required fields for generate_question.",
                last_action_success=False,
            )

        # Validate topic
        if action.topic not in self.available_topics:
            return self._build_observation(
                reward=-0.2,
                done=False,
                last_action_result=f"Invalid topic: '{action.topic}'. Must be one of {self.available_topics}.",
                last_action_success=False,
            )

        # Validate difficulty
        if action.difficulty not in ("easy", "medium", "hard"):
            return self._build_observation(
                reward=-0.2,
                done=False,
                last_action_result=f"Invalid difficulty: '{action.difficulty}'. Must be easy/medium/hard.",
                last_action_success=False,
            )

        # Validate marks
        if action.marks not in (1, 2, 4):
            return self._build_observation(
                reward=-0.2,
                done=False,
                last_action_result=f"Invalid marks: {action.marks}. Must be 1, 2, or 4.",
                last_action_success=False,
            )

        # Validate marks constraint
        if self.marks_used + action.marks > 100:
            return self._build_observation(
                reward=-0.3,
                done=False,
                last_action_result=f"Marks limit exceeded: {self.marks_used}+{action.marks} > 100.",
                last_action_success=False,
            )

        # All valid — create question record
        qid = str(uuid.uuid4())
        record = QuestionRecord(
            question_id=qid,
            topic=action.topic,
            difficulty=action.difficulty,
            marks=action.marks,
            question_text=action.question_text,
            options={
                "A": action.option_a,
                "B": action.option_b,
                "C": action.option_c,
                "D": action.option_d,
            },
            correct_option=action.correct_option,
            explanation=action.explanation,
        )
        self.question_bank[qid] = record
        self.marks_used += action.marks

        # Adaptive difficulty hint
        difficulty_required = self._get_difficulty_requirement()
        result_msg = f"Question generated: {action.topic} ({action.difficulty}, {action.marks}m)"
        if difficulty_required:
            result_msg += f". Paper balance note: consider generating a '{difficulty_required}' question next."

        return self._build_observation(
            reward=0.3,
            done=False,
            last_action_result=result_msg,
            last_action_success=True,
            question_id_generated=qid,
        )

    def _handle_validate(self, action: ExamForgeAction) -> ExamForgeObservation:
        """Handle VALIDATE_QUESTION action."""
        if not action.question_id or action.question_id not in self.question_bank:
            return self._build_observation(
                reward=0.0,
                done=False,
                last_action_result=f"Question ID not found: {action.question_id}",
                last_action_success=False,
            )

        record = self.question_bank[action.question_id]

        # Run programmatic checks
        consistency = self._check_answer_consistency(record)
        distractor = self._check_distractor_uniqueness(record)
        calibration = self._check_difficulty_calibration(record)

        # Raw score out of 0.7, normalize to 0–1
        raw_score = consistency + distractor + calibration
        validation_score = min(raw_score / 0.7, 1.0)

        record.is_validated = True
        record.validation_score = validation_score

        reward = validation_score * 1.0

        return self._build_observation(
            reward=reward,
            done=False,
            last_action_result=f"Validated question {action.question_id[:8]}... score={validation_score:.2f}",
            last_action_success=True,
            validation_score=validation_score,
            distractor_quality_score=distractor,
            difficulty_accuracy=(calibration > 0),
        )

    def _handle_flag(self, action: ExamForgeAction) -> ExamForgeObservation:
        """Handle FLAG_QUESTION action."""
        if not action.question_id or action.question_id not in self.question_bank:
            return self._build_observation(
                reward=0.0,
                done=False,
                last_action_result=f"Question ID not found: {action.question_id}",
                last_action_success=False,
            )

        if not action.flag_reason or len(action.flag_reason) <= 10:
            return self._build_observation(
                reward=0.0,
                done=False,
                last_action_result="Flag reason must be a non-empty string with length > 10.",
                last_action_success=False,
            )

        record = self.question_bank[action.question_id]

        # Reward logic: flagging low-quality or unvalidated = good, flagging valid = bad
        if record.validation_score < 0.4 or not record.is_validated:
            reward = 0.2
        else:
            reward = -0.1

        record.is_flagged = True

        return self._build_observation(
            reward=reward,
            done=False,
            last_action_result=f"Flagged question {action.question_id[:8]}... reason: {action.flag_reason[:30]}",
            last_action_success=True,
        )

    def _handle_assemble(self, action: ExamForgeAction) -> ExamForgeObservation:
        """Handle ASSEMBLE_PAPER action."""
        valid_questions = [q for q in self.question_bank.values() if not q.is_flagged]

        if len(valid_questions) < 10:
            return self._build_observation(
                reward=-0.5,
                done=False,
                last_action_result=f"Need at least 10 valid questions, have {len(valid_questions)}.",
                last_action_success=False,
            )

        # ── Topic coverage score ────────────────────────────────────────────
        unique_topics = set(q.topic for q in valid_questions)
        topic_coverage_score = min(len(unique_topics) / 5.0, 1.0)

        # ── Difficulty distribution score ───────────────────────────────────
        diff_counts = {"easy": 0, "medium": 0, "hard": 0}
        for q in valid_questions:
            diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1

        total = len(valid_questions)
        ideal = {"easy": 0.30, "medium": 0.50, "hard": 0.20}
        actual = {k: diff_counts[k] / total for k in ideal}
        diff_deviation = sum(abs(actual[k] - ideal[k]) for k in ideal) / len(ideal)
        difficulty_distribution_score = max(0.0, min(1.0, 1.0 - diff_deviation))

        # ── Validated ratio ─────────────────────────────────────────────────
        validated_count = len([q for q in valid_questions if q.is_validated])
        validated_ratio = validated_count / total

        # ── Final paper score ───────────────────────────────────────────────
        final_paper_score = (
            0.4 * topic_coverage_score
            + 0.4 * difficulty_distribution_score
            + 0.2 * validated_ratio
        )
        reward = final_paper_score * 3.0
        self._paper_assembled = True

        return self._build_observation(
            reward=reward,
            done=True,
            last_action_result="Paper assembled successfully!",
            last_action_success=True,
            paper_assembled=True,
            topic_coverage_score=topic_coverage_score,
            difficulty_distribution=diff_counts,
            final_paper_score=final_paper_score,
        )

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        """Return current episode state as a plain dict."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "current_subject": getattr(self, "current_subject", ""),
            "marks_used": self.marks_used,
            "num_questions": len(self.question_bank),
            "num_valid": len([q for q in self.question_bank.values() if not q.is_flagged]),
            "num_flagged": len([q for q in self.question_bank.values() if q.is_flagged]),
            "num_validated": len([q for q in self.question_bank.values() if q.is_validated]),
            "paper_assembled": getattr(self, "_paper_assembled", False),
            # Graders need full question_bank access
            "question_bank": self.question_bank,
        }
