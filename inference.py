"""
ExamForge Demo Agent — inference.py

Demonstrates a complete episode WITHOUT connecting to a live server.
Tests the environment logic directly by generating, validating, and
assembling a JEE Physics exam paper with 15 hardcoded realistic MCQs.
"""

import sys
sys.path.insert(0, ".")

from server.environment import ExamForgeEnvironment, SUBJECT_TOPICS
from models import ExamForgeAction, ActionType


# ─────────────────────────────────────────────────────────────────────────────
# 15 Realistic JEE Physics MCQs
# Covering ≥5 topics, mix of easy (5), medium (7), hard (3)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    # ── EASY (5 questions, 1 mark each) ──────────────────────────────────────
    {
        "topic": "Kinematics",
        "difficulty": "easy",
        "marks": 1,
        "question_text": "A body is thrown vertically upward with velocity u. The greatest height h to which it will rise is:",
        "option_a": "u / 2g",
        "option_b": "u² / 2g",
        "option_c": "u² / g",
        "option_d": "u / g",
        "correct_option": "B",
        "explanation": "Using v² = u² − 2gh, at greatest height v = 0, so h = u²/2g. Option B is correct.",
    },
    {
        "topic": "Laws of Motion",
        "difficulty": "easy",
        "marks": 1,
        "question_text": "Newton's first law of motion defines which physical quantity?",
        "option_a": "Velocity",
        "option_b": "Force",
        "option_c": "Inertia",
        "option_d": "Momentum",
        "correct_option": "C",
        "explanation": "Newton's first law states that a body continues in its state of rest or uniform motion unless acted upon by an external force. This law defines inertia. Option C is correct.",
    },
    {
        "topic": "Optics",
        "difficulty": "easy",
        "marks": 1,
        "question_text": "The image formed by a convex mirror is always:",
        "option_a": "Real and inverted",
        "option_b": "Virtual, erect, and diminished",
        "option_c": "Virtual and magnified",
        "option_d": "Real and magnified",
        "correct_option": "B",
        "explanation": "A convex mirror always produces a virtual, erect, and diminished image regardless of the object distance. This is because the focus and centre of curvature are behind the mirror. Option B is correct.",
    },
    {
        "topic": "Modern Physics",
        "difficulty": "easy",
        "marks": 1,
        "question_text": "The photoelectric effect demonstrates the:",
        "option_a": "Wave nature of light",
        "option_b": "Particle nature of light",
        "option_c": "Dual nature of matter",
        "option_d": "Wave-particle duality of electrons",
        "correct_option": "B",
        "explanation": "The photoelectric effect was explained by Einstein using the photon concept — light consists of discrete energy packets (photons). This demonstrates the particle nature of light. Option B is correct.",
    },
    {
        "topic": "Current Electricity",
        "difficulty": "easy",
        "marks": 1,
        "question_text": "The SI unit of electrical resistance is:",
        "option_a": "Ampere",
        "option_b": "Volt",
        "option_c": "Ohm",
        "option_d": "Watt",
        "correct_option": "C",
        "explanation": "Electrical resistance is measured in ohms (Ω), named after Georg Simon Ohm. One ohm equals one volt per ampere. Option C is correct.",
    },

    # ── MEDIUM (7 questions, 2 marks each) ───────────────────────────────────
    {
        "topic": "Work & Energy",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "A force F = (3î + 4ĵ) N acts on a body and displaces it by s = (3î + 4ĵ) m. The work done is:",
        "option_a": "10 J",
        "option_b": "15 J",
        "option_c": "25 J",
        "option_d": "20 J",
        "correct_option": "C",
        "explanation": "Work done W = F · s = (3×3) + (4×4) = 9 + 16 = 25 J. The dot product of force and displacement vectors gives the scalar work done. Option C is correct.",
    },
    {
        "topic": "Thermodynamics",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "In an isothermal process for an ideal gas, which quantity remains constant?",
        "option_a": "Pressure",
        "option_b": "Volume",
        "option_c": "Temperature",
        "option_d": "Entropy",
        "correct_option": "C",
        "explanation": "By definition, an isothermal process occurs at constant temperature. For an ideal gas undergoing isothermal change, PV = nRT remains constant since T is constant. Option C is correct.",
    },
    {
        "topic": "Electrostatics",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "Two point charges +q and −q are placed at distance d apart. The electric field at the midpoint of the line joining them is:",
        "option_a": "Zero",
        "option_b": "kq/d² directed from +q to −q",
        "option_c": "4kq/d² directed from +q to −q",
        "option_d": "8kq/d² directed from +q to −q",
        "correct_option": "D",
        "explanation": "At the midpoint, each charge is at distance d/2. E from +q = kq/(d/2)² = 4kq/d² pointing away, E from −q = 4kq/d² pointing towards −q. Both point in same direction (from +q to −q), so net E = 8kq/d². Option D is correct.",
    },
    {
        "topic": "Magnetism",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "A charged particle moving with velocity v enters a uniform magnetic field B perpendicular to its motion. The path of the particle is:",
        "option_a": "Straight line",
        "option_b": "Parabola",
        "option_c": "Circle",
        "option_d": "Ellipse",
        "correct_option": "C",
        "explanation": "When a charged particle enters a uniform magnetic field perpendicular to its velocity, the magnetic force qv×B acts as centripetal force, causing circular motion. The radius is r = mv/qB. Option C is correct.",
    },
    {
        "topic": "Kinematics",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "A projectile is launched at angle θ with horizontal with speed u. The time of flight is:",
        "option_a": "u sin θ / g",
        "option_b": "2u sin θ / g",
        "option_c": "u cos θ / g",
        "option_d": "2u cos θ / g",
        "correct_option": "B",
        "explanation": "The time of flight T = 2u sinθ/g. This comes from analyzing vertical motion: at the highest point vy = 0, so time to reach top = u sinθ/g, and total flight time is twice that. Option B is correct.",
    },
    {
        "topic": "Rotational Motion",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "The moment of inertia of a solid sphere of mass M and radius R about its diameter is:",
        "option_a": "MR²",
        "option_b": "2MR²/3",
        "option_c": "2MR²/5",
        "option_d": "MR²/2",
        "correct_option": "C",
        "explanation": "The moment of inertia of a solid sphere about its diameter is (2/5)MR². This is derived by integrating dm × r² over the volume using thin disk elements. Option C is correct.",
    },
    {
        "topic": "Work & Energy",
        "difficulty": "medium",
        "marks": 2,
        "question_text": "A spring of spring constant k is compressed by distance x. The potential energy stored is:",
        "option_a": "kx",
        "option_b": "kx²",
        "option_c": "kx²/2",
        "option_d": "2kx²",
        "correct_option": "C",
        "explanation": "The elastic potential energy stored in a spring is given by U = ½kx², where k is the spring constant and x is the compression or extension from natural length. Option C is correct.",
    },

    # ── HARD (3 questions, 4 marks each) ─────────────────────────────────────
    {
        "topic": "Thermodynamics",
        "difficulty": "hard",
        "marks": 4,
        "question_text": "One mole of an ideal monoatomic gas undergoes an adiabatic process where temperature changes from T₁ to T₂. The work done by the gas is:",
        "option_a": "nCv(T₁ − T₂)",
        "option_b": "nCp(T₁ − T₂)",
        "option_c": "nR(T₁ − T₂)/(γ − 1)",
        "option_d": "Both A and C",
        "correct_option": "D",
        "explanation": "For an adiabatic process, Q = 0, so by first law W = −ΔU = nCv(T₁ − T₂). Also W = nR(T₁ − T₂)/(γ − 1). Since Cv = R/(γ − 1) for ideal gas, both expressions are equivalent. For monoatomic gas, γ = 5/3, Cv = 3R/2. So W = (3/2)nR(T₁ − T₂). Option D is correct because both A and C give the same result.",
    },
    {
        "topic": "Electrostatics",
        "difficulty": "hard",
        "marks": 4,
        "question_text": "A charge Q is distributed uniformly over a thin ring of radius R. The electric potential at a point P on the axis at distance x from the centre is:",
        "option_a": "kQ/x",
        "option_b": "kQ/R",
        "option_c": "kQ/√(R² + x²)",
        "option_d": "kQx/(R² + x²)",
        "correct_option": "C",
        "explanation": "Every element dq on the ring is at the same distance √(R² + x²) from point P on the axis. Since potential is a scalar, V = ∫k dq/√(R² + x²) = kQ/√(R² + x²). This is a fundamental result in electrostatics. At x = 0 (centre), V = kQ/R, and as x → ∞, V → kQ/x (like a point charge). Option C is correct.",
    },
    {
        "topic": "Modern Physics",
        "difficulty": "hard",
        "marks": 4,
        "question_text": "In hydrogen atom, the ratio of the frequencies of the first line of the Lyman series to the first line of the Balmer series is:",
        "option_a": "27/5",
        "option_b": "5/27",
        "option_c": "27/8",
        "option_d": "8/27",
        "correct_option": "A",
        "explanation": "First Lyman line: 1/λ₁ = R(1/1² − 1/2²) = R(3/4), so ν₁ = Rc(3/4). First Balmer line: 1/λ₂ = R(1/2² − 1/3²) = R(5/36), so ν₂ = Rc(5/36). Ratio = (3/4)/(5/36) = (3/4)(36/5) = 108/20 = 27/5. This requires careful application of the Rydberg formula for hydrogen spectral series. Option A is correct.",
    },
]

assert len(SAMPLE_QUESTIONS) == 15, f"Expected 15 questions, got {len(SAMPLE_QUESTIONS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo Episode Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demo_episode():
    """Run a complete ExamForge episode using the hardcoded JEE Physics MCQs."""
    env = ExamForgeEnvironment()

    # Force subject to "JEE Physics" for reproducibility
    env.current_subject = "JEE Physics"
    env.available_topics = list(SUBJECT_TOPICS["JEE Physics"])
    env.paper_constraints = {"total_marks": 100, "num_questions": 25, "time_limit_mins": 180}
    env.question_bank = {}
    env.step_count = 0
    env.marks_used = 0
    env.episode_id = "demo-episode-001"

    total_reward = 0.0
    print("=" * 70)
    print("ExamForge Demo Agent — JEE Physics Episode")
    print("=" * 70)
    print(f"Episode started | Subject: JEE Physics")
    print(f"Available topics: {env.available_topics}")
    print(f"Paper constraints: {env.paper_constraints}")
    print("-" * 70)

    task_name = "examforge_demo"
    print(f"[START] task={task_name}", flush=True)
    step_counter = 0

    generated_ids = []

    # Phase 1: Generate and validate all 15 questions
    for i, q in enumerate(SAMPLE_QUESTIONS):
        # Step 1: Generate question
        action = ExamForgeAction(
            action_type=ActionType.GENERATE_QUESTION,
            topic=q["topic"],
            difficulty=q["difficulty"],
            marks=q["marks"],
            question_text=q["question_text"],
            option_a=q["option_a"],
            option_b=q["option_b"],
            option_c=q["option_c"],
            option_d=q["option_d"],
            correct_option=q["correct_option"],
            explanation=q["explanation"],
        )
        obs = env.step(action)
        step_counter += 1
        print(f"[STEP] step={step_counter} reward={obs.reward}", flush=True)
        total_reward += obs.reward
        qid = obs.question_id_generated
        generated_ids.append(qid)
        print(f"Generated Q{i+1:2d}: {q['topic']:20s} ({q['difficulty']:6s}, {q['marks']}m) → reward: {obs.reward:+.2f}")

        # Step 2: Validate the question
        val_action = ExamForgeAction(
            action_type=ActionType.VALIDATE_QUESTION,
            question_id=qid,
        )
        obs = env.step(val_action)
        step_counter += 1
        print(f"[STEP] step={step_counter} reward={obs.reward}", flush=True)
        total_reward += obs.reward
        print(f"  Validated  → score: {obs.validation_score:.2f}, reward: {obs.reward:+.2f}")

    print("-" * 70)

    # Phase 2: Flag low-quality questions (validation_score < 0.4)
    flagged_count = 0
    for qid in generated_ids:
        record = env.question_bank[qid]
        if record.is_validated and record.validation_score < 0.4:
            flag_action = ExamForgeAction(
                action_type=ActionType.FLAG_QUESTION,
                question_id=qid,
                flag_reason="Low validation score — question quality below threshold for exam inclusion.",
            )
            obs = env.step(flag_action)
            step_counter += 1
            print(f"[STEP] step={step_counter} reward={obs.reward}", flush=True)
            total_reward += obs.reward
            flagged_count += 1
            print(f"  Flagged Q (score={record.validation_score:.2f}) → reward: {obs.reward:+.2f}")

    if flagged_count == 0:
        print("  No questions flagged — all passed quality threshold!")

    print("-" * 70)

    # Phase 3: Assemble paper
    assemble_action = ExamForgeAction(action_type=ActionType.ASSEMBLE_PAPER)
    obs = env.step(assemble_action)
    step_counter += 1
    print(f"[STEP] step={step_counter} reward={obs.reward}", flush=True)
    total_reward += obs.reward

    print()
    print("=" * 70)
    print("=== PAPER ASSEMBLED ===" if obs.paper_assembled else "=== ASSEMBLY FAILED ===")
    print("=" * 70)
    print(f"Final Paper Score:       {obs.final_paper_score:.3f}")
    print(f"Topic Coverage:          {obs.topic_coverage_score:.3f}")
    print(f"Difficulty Distribution: {obs.difficulty_distribution}")
    print(f"Total Reward Assembly:   {obs.reward:+.2f}")
    print(f"Total Steps Used:        {env.step_count}")
    print(f"Total Marks Used:        {env.marks_used}")
    print(f"Questions in Bank:       {len(env.question_bank)}")
    print(f"Valid (unflagged):       {len([q for q in env.question_bank.values() if not q.is_flagged])}")
    print("-" * 70)
    print(f"TOTAL ACCUMULATED REWARD: {total_reward:+.2f}")
    print("=" * 70)

    print(f"[END] task={task_name} score={obs.final_paper_score} steps={step_counter}", flush=True)

    return total_reward


if __name__ == "__main__":
    run_demo_episode()
