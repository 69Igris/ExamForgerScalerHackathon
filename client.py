"""
ExamForge Client — convenience wrapper for EnvClient.

Provides typed convenience methods for each action type so agents
can interact with the environment using simple Python calls.
"""

from openenv.core import EnvClient
from models import ExamForgeAction, ExamForgeObservation, ActionType


class ExamForgeEnv(EnvClient):
    """
    Client-side connector for the ExamForge environment.

    Usage (async):
        async with ExamForgeEnv(base_url="ws://localhost:7860") as env:
            obs = await env.reset()
            obs = await env.generate_question(...)

    Usage (sync):
        env = ExamForgeEnv(base_url="ws://localhost:7860").sync()
        with env:
            obs = env.reset()
            obs = env.generate_question(...)
    """
    action_class = ExamForgeAction
    observation_class = ExamForgeObservation

    async def generate_question(self, topic, difficulty, marks, question_text,
                                option_a, option_b, option_c, option_d,
                                correct_option, explanation):
        """Convenience method to generate a question."""
        return await self.step(ExamForgeAction(
            action_type=ActionType.GENERATE_QUESTION,
            topic=topic, difficulty=difficulty, marks=marks,
            question_text=question_text,
            option_a=option_a, option_b=option_b,
            option_c=option_c, option_d=option_d,
            correct_option=correct_option,
            explanation=explanation,
        ))

    async def validate_question(self, question_id: str):
        """Convenience method to validate a question."""
        return await self.step(ExamForgeAction(
            action_type=ActionType.VALIDATE_QUESTION,
            question_id=question_id,
        ))

    async def flag_question(self, question_id: str, reason: str):
        """Convenience method to flag a question."""
        return await self.step(ExamForgeAction(
            action_type=ActionType.FLAG_QUESTION,
            question_id=question_id,
            flag_reason=reason,
        ))

    async def assemble_paper(self):
        """Convenience method to finalize and assemble the paper."""
        return await self.step(ExamForgeAction(
            action_type=ActionType.ASSEMBLE_PAPER,
        ))
