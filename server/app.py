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

    return app


# Create the app instance (imported by uvicorn)
app = create_examforge_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
