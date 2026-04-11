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
        """
        Returns the task catalog for automated evaluation.
        Required by OpenEnv spec for task discovery.
        """
        return JSONResponse({
            "tasks": [
                {
                    "name": "question_generation",
                    "description": (
                        "EASY: Agent generates 12+ high-quality MCQs across 5+ topics "
                        "for Indian competitive exams (JEE, GATE, UPSC). Graded on question "
                        "count, topic diversity, and marks variety."
                    ),
                    "difficulty": "easy",
                    "grader": "server.environment:question_generation_grader",
                    "max_steps": 20,
                    "expected_score_range": [0.60, 0.99],
                },
                {
                    "name": "question_validation",
                    "description": (
                        "MEDIUM: Agent generates questions then validates each one using "
                        "programmatic quality checks (answer consistency, distractor quality, "
                        "difficulty calibration). Flags low-quality items."
                    ),
                    "difficulty": "medium",
                    "grader": "server.environment:question_validation_grader",
                    "max_steps": 40,
                    "expected_score_range": [0.50, 0.90],
                },
                {
                    "name": "paper_assembly",
                    "description": (
                        "HARD: Full pipeline — generate, validate, flag, then assemble a "
                        "complete balanced exam paper meeting topic coverage (5+ topics) and "
                        "difficulty distribution targets (30% easy, 50% medium, 20% hard)."
                    ),
                    "difficulty": "hard",
                    "grader": "server.environment:paper_assembly_grader",
                    "max_steps": 50,
                    "expected_score_range": [0.40, 0.85],
                },
            ]
        })

    # Grader endpoint for remote scoring
    @app.post("/grader")
    async def run_grader(request: dict = None):
        """
        Grades a task given episode state or episode parameters.

        Request body (all optional):
        {
            "task_name": "question_generation" | "question_validation" | "paper_assembly",
            "episode_id": "optional-existing-episode-id",
            "question_bank": {},
            "paper_assembled": false,
            "marks_used": 0
        }

        Returns: {"task_name": "...", "score": 0.0-1.0, "grader_used": "..."}
        """
        from server.environment import (
            question_generation_grader,
            question_validation_grader,
            paper_assembly_grader,
        )

        if request is None:
            request = {}

        task_name = request.get("task_name", "paper_assembly")

        # Build episode state for grading
        episode_state = {
            "question_bank": request.get("question_bank", {}),
            "paper_assembled": request.get("paper_assembled", False),
            "marks_used": request.get("marks_used", 0),
            "step_count": request.get("step_count", 0),
        }

        grader_map = {
            "question_generation": (
                question_generation_grader,
                "server.environment:question_generation_grader"
            ),
            "question_validation": (
                question_validation_grader,
                "server.environment:question_validation_grader"
            ),
            "paper_assembly": (
                paper_assembly_grader,
                "server.environment:paper_assembly_grader"
            ),
        }

        if task_name not in grader_map:
            return JSONResponse(
                {"error": f"Unknown task: {task_name}. Valid tasks: {list(grader_map.keys())}"},
                status_code=400
            )

        grader_fn, grader_path = grader_map[task_name]

        try:
            score = grader_fn(episode_state)
            score = float(min(max(score, 0.0), 1.0))
            return JSONResponse({
                "task_name": task_name,
                "score": score,
                "grader_used": grader_path,
            })
        except Exception as e:
            print(f"[DEBUG] Grader error: {e}", flush=True)
            return JSONResponse({
                "task_name": task_name,
                "score": 0.0,
                "grader_used": grader_path,
                "error": str(e),
            }, status_code=500)

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
