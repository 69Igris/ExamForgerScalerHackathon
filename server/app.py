"""
ExamForge FastAPI Server — serves the environment via OpenEnv's HTTP/WS server.
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core import create_app
from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ExamForgeObservation

app = create_app(
    env=lambda: ExamForgeEnvironment(),
    action_cls=ExamForgeAction,
    observation_cls=ExamForgeObservation,
    env_name="examforge_env",
    max_concurrent_envs=32,
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
