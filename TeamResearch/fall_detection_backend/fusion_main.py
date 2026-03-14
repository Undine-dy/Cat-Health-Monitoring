import logging

from fastapi import FastAPI, HTTPException

from fusion_engine import FusionEngine
from fusion_schemas import SensorObservation
from project_config import FUSION_HOST, FUSION_MODEL_DIR, FUSION_PORT


ACTIVITY_MODEL_PATH = FUSION_MODEL_DIR / "activity_context_model.joblib"
HR_REFERENCE_PATH = FUSION_MODEL_DIR / "hr_reference.json"
STRESS_MODEL_PATH = FUSION_MODEL_DIR / "stress_classifier.joblib"

logger = logging.getLogger("FusionBackend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

engine: FusionEngine | None = None


def build_engine() -> FusionEngine:
    if not ACTIVITY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing activity model: {ACTIVITY_MODEL_PATH}")
    if not HR_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Missing HR reference: {HR_REFERENCE_PATH}")
    return FusionEngine(
        activity_model_path=ACTIVITY_MODEL_PATH,
        hr_reference_path=HR_REFERENCE_PATH,
        stress_model_path=STRESS_MODEL_PATH if STRESS_MODEL_PATH.exists() else None,
    )


app = FastAPI(title="Wearable Fusion Backend", description="跌倒项目升级版：活动上下文 + 压力风险 + 心率异常")


@app.on_event("startup")
async def on_startup() -> None:
    global engine
    engine = build_engine()
    logger.info("融合系统启动完成")


@app.get("/")
async def read_root() -> dict:
    return {
        "status": "running",
        "service": "wearable-fusion-backend",
        "models_loaded": bool(engine),
    }


@app.post("/fusion/ingest")
async def fusion_ingest(observation: SensorObservation) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Fusion engine not ready")
    return engine.ingest(observation)


@app.get("/fusion/state/{user_id}")
async def fusion_state(user_id: str) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Fusion engine not ready")
    state = engine.get_state(user_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    return state


@app.get("/fusion/alerts/{user_id}")
async def fusion_alerts(user_id: str) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Fusion engine not ready")
    return {"user_id": user_id, "alerts": engine.get_alerts(user_id)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=FUSION_HOST, port=FUSION_PORT, log_level="info")
