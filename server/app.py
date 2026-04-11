"""
ExamForge — FastAPI Application Server
=======================================
Wraps ExamForgeEnvironment in an OpenEnv-compliant FastAPI server.
Runs on port 7860 for HuggingFace Spaces compatibility.
"""
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Import openenv server factory
try:
    from openenv.core import create_app
    _USE_OPENENV = True
except ImportError:
    _USE_OPENENV = False

from server.environment import ExamForgeEnvironment
from models import ExamForgeAction, ExamForgeObservation


def create_examforge_app() -> FastAPI:
    """Create and configure the ExamForge FastAPI application."""
    
    if _USE_OPENENV:
        app = create_app(
            env=lambda: ExamForgeEnvironment(),
            action_cls=ExamForgeAction,
            observation_cls=ExamForgeObservation,
            env_name="examforge_env",
            max_concurrent_envs=32,
        )
    else:
        # Fallback: minimal FastAPI app with manual endpoints
        app = FastAPI(
            title="ExamForge OpenEnv",
            description="RL environment for competitive exam question generation",
            version="1.0.0",
        )
        _env_instance = ExamForgeEnvironment()

        @app.post("/reset")
        async def reset():
            obs = _env_instance.reset()
            return obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        @app.post("/step")
        async def step(action: ExamForgeAction):
            result = _env_instance.step(action)
            return result.model_dump() if hasattr(result, "model_dump") else result.__dict__

        @app.get("/state")
        async def state():
            return _env_instance.state()

    # Add /health endpoint (always present, regardless of openenv)
    @app.get("/health")
    async def health():
        """Health check endpoint — required by HuggingFace Spaces."""
        return JSONResponse({"status": "healthy", "env": "examforge", "version": "1.0.0"})

    # Add root endpoint for browser access
    @app.get("/")
    async def root():
        return JSONResponse({
            "name": "ExamForge",
            "description": "OpenEnv RL environment for competitive exam question generation",
            "tasks": ["question_generation", "question_validation", "paper_assembly"],
            "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
        })

    # Task catalog endpoint for automated evaluation
    @app.get("/tasks")
    async def list_tasks():
        """Return the task catalog for automated evaluation."""
        return {
            "tasks": [
                {
                    "name": "question_generation",
                    "description": "Generate high-quality MCQ questions across multiple topics. Easy difficulty.",
                    "difficulty": "easy",
                    "grader": "server.environment:question_generation_grader",
                    "max_steps": 20
                },
                {
                    "name": "question_validation",
                    "description": "Generate, validate, and flag questions. Medium difficulty.",
                    "difficulty": "medium",
                    "grader": "server.environment:question_validation_grader",
                    "max_steps": 40
                },
                {
                    "name": "paper_assembly",
                    "description": "Full pipeline: generate, validate, flag, assemble a balanced paper. Hard difficulty.",
                    "difficulty": "hard",
                    "grader": "server.environment:paper_assembly_grader",
                    "max_steps": 50
                }
            ]
        }

    # Grader endpoint for remote scoring
    @app.post("/grader")
    async def run_grader(request: dict = None):
        """
        Grade a task given episode state or episode_id.
        Accepts: {"episode_id": "...", "task_name": "..."}
              or {"task_name": "...", "question_bank": {...}, "paper_assembled": bool}
        Returns: {"task_name": "...", "score": 0.0-1.0}
        """
        from server.environment import (
            question_generation_grader,
            question_validation_grader,
            paper_assembly_grader
        )

        if request is None:
            request = {}

        task_name = request.get("task_name", "paper_assembly")

        # Build episode state from request or use empty state
        episode_state = {
            "question_bank": request.get("question_bank", {}),
            "paper_assembled": request.get("paper_assembled", False),
            "marks_used": request.get("marks_used", 0),
        }

        grader_map = {
            "question_generation": question_generation_grader,
            "question_validation": question_validation_grader,
            "paper_assembly": paper_assembly_grader,
        }

        grader_fn = grader_map.get(task_name, paper_assembly_grader)

        try:
            score = grader_fn(episode_state)
            score = float(min(max(score, 0.0), 1.0))
        except Exception as e:
            print(f"[DEBUG] Grader error: {e}", flush=True)
            score = 0.0

        return {"task_name": task_name, "score": score}

    return app


# Create the app instance (imported by uvicorn)
app = create_examforge_app()


def main():
    """Start the ExamForge server — required for multi-mode deployment."""
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
