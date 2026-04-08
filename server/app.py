"""
MedAuditEnv FastAPI Server (package entrypoint)
Provides HTTP endpoints for the environment.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from medaudit import MedAuditAction, MedAuditEnv

app = FastAPI(
    title="MedAuditEnv",
    description="Rural Medical Record Auditor - OpenEnv for fraud detection",
    version="1.0.0",
)

# Global environment instance (single-user for demo)
env_instance: MedAuditEnv | None = None


class ResetRequest(BaseModel):
    """Request to reset environment"""

    task: str = "vital_check"  # Options: vital_check, fraud_mix, batch_audit


class StepRequest(BaseModel):
    """Request to step environment"""

    action: str  # One of: flag_anomaly, approve_claim, reject_claim, request_clarification


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": "MedAuditEnv",
        "version": "1.0.0",
        "tasks": ["vital_check", "fraud_mix", "batch_audit"],
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset environment for new episode and return initial observation."""
    global env_instance

    try:
        env_instance = MedAuditEnv(task=request.task, data_path="data/claims.json")
        observation = env_instance.reset()
        return JSONResponse(
            content={"observation": observation.model_dump(), "state": env_instance.state().model_dump()}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Claims data not found. Run data_generator.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Execute action in environment."""
    global env_instance

    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        try:
            action = MedAuditAction(request.action)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. Must be one of: {[a.value for a in MedAuditAction]}",
            )

        result = env_instance.step(action)
        return JSONResponse(
            content={
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
                "state": env_instance.state().model_dump(),
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state."""
    global env_instance

    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    return JSONResponse(content=env_instance.state().model_dump())


@app.get("/score")
async def get_score():
    """Get normalized score for completed episode."""
    global env_instance

    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    if not env_instance.state().done:
        raise HTTPException(status_code=400, detail="Episode not complete. Finish all steps first.")

    score = env_instance.calculate_score()
    return JSONResponse(
        content={
            "score": score,
            "task": env_instance.task,
            "steps": len(env_instance.step_rewards),
            "accuracy": env_instance.state().accuracy,
        }
    )


@app.get("/tasks")
async def list_tasks():
    """List available tasks."""
    return JSONResponse(
        content={
            "tasks": [
                {
                    "name": "vital_check",
                    "difficulty": "easy",
                    "claims": 10,
                    "description": "Detect impossible vital signs",
                },
                {
                    "name": "fraud_mix",
                    "difficulty": "medium",
                    "claims": 20,
                    "description": "Mixed fraud pattern detection",
                },
                {
                    "name": "batch_audit",
                    "difficulty": "hard",
                    "claims": 50,
                    "description": "Full batch fraud audit",
                },
            ]
        }
    )


def main() -> None:
    """Console script entrypoint (for `[project.scripts]`)."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

