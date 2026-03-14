import re
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import joblib
import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from project_config import BACKEND_DIR, LEGACY_HOST, LEGACY_PORT, MODEL_DIR, UCI_HAR_ROOT
from wellness_service import (
    WellnessChatRequest,
    WellnessReportRequest,
    build_formal_record_file,
    chat_with_session,
    close_wellness_service,
    create_wellness_report,
    load_demo_user,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FallDetectionAPI")

class Settings(BaseSettings):
    HOST: str = LEGACY_HOST
    PORT: int = LEGACY_PORT
    LOG_LEVEL: str = "INFO"

    BASE_DIR: Path = BACKEND_DIR
    MODEL_PATH: Path = MODEL_DIR / "svm_baseline.joblib"
    ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"
    FEATURES_PATH: Path = MODEL_DIR / "feature_names.pkl"
    ACTIVITY_LABELS_PATH: Path = UCI_HAR_ROOT / "activity_labels.txt"

    QWEN_API_KEY: str = Field(default="", env="QWEN_API_KEY")
    QWEN_API_URL: str = Field(default="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation", env="QWEN_API_URL")
    QWEN_MODEL: str = Field(default="qwen-plus", env="QWEN_MODEL")

    ACTION_THRESHOLD: int = 3
    SUSPECT_TIMEOUT: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()
UI_DIR = BACKEND_DIR / "ui"
ASSETS_DIR = BACKEND_DIR / "assets"

class ActionPrediction(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    activity_label: str = Field(..., description="活动标签，如 WALKING, LAYING")
    timestamp: datetime = Field(default_factory=datetime.now)

class UserReply(BaseModel):
    user_id: str
    reply_text: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SensorData(BaseModel):
    user_id: str
    features: list[float] = Field(..., min_length=561, max_length=561, description="561 维传感器特征")
    timestamp: datetime = Field(default_factory=datetime.now)

class FallDetectionRequest(BaseModel):
    user_id: str
    motion_data: Dict[str, Any]

class UserState:
    def __init__(self):
        self.current_state = "normal"
        self.last_activity: Optional[str] = None
        self.activity_count: int = 0
        self.last_effective_activity: Optional[str] = None
        self.suspect_start_time: Optional[datetime] = None
        self.last_question: Optional[str] = None
        self.last_emergency_msg: Optional[str] = None

class StateManager:
    def __init__(self):
        self._states: Dict[str, UserState] = {}
        self._lock = asyncio.Lock()
        self.states_map = {
            "normal": "normal",
            "suspect_fall": "suspect_fall",
            "confirmed_fall": "confirmed_fall"
        }

    async def get_or_create(self, user_id: str) -> UserState:
        async with self._lock:
            if user_id not in self._states:
                self._states[user_id] = UserState()
                logger.info(f"初始化用户状态：{user_id}")
            return self._states[user_id]

    async def get_status(self, user_id: str) -> Optional[dict]:
        async with self._lock:
            if user_id in self._states:
                state = self._states[user_id]
                return {
                    "current_state": state.current_state,
                    "last_activity": state.last_activity,
                    "suspect_start_time": state.suspect_start_time.isoformat() if state.suspect_start_time else None,
                    "last_question": state.last_question,
                    "last_emergency_msg": state.last_emergency_msg
                }
            return None
    
    async def update_state(self, user_id: str, new_state: str, reset_suspect_time: bool = False):
        async with self._lock:
            if user_id in self._states:
                state = self._states[user_id]
                state.current_state = new_state
                if reset_suspect_time:
                    state.suspect_start_time = None
                logger.info(f"用户 {user_id} 状态更新为：{new_state}")

state_manager = StateManager()

class LLMService:
    def __init__(self, api_key: str, api_url: str, model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def generate(self, prompt: str, system_message: str = "你是一个智能助手。") -> Optional[str]:
        if not self.api_key or self.api_key == "your-api-key-here":
            logger.warning("LLM API Key 未配置，跳过调用")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {"temperature": 0.7, "max_tokens": 150, "top_p": 0.8}
        }

        try:
            response = await self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            text = ""
            if "output" in result and "text" in result["output"]:
                text = result["output"]["text"]
            elif "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("message", {}).get("content", "")
            
            return text.strip() if text else None
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP 错误：{e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"LLM 调用异常：{e}")
            return None

llm_service = LLMService(settings.QWEN_API_KEY, settings.QWEN_API_URL, settings.QWEN_MODEL)

class ModelManager:
    def __init__(self):
        self.model: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.feature_names: Optional[list] = None
        self.activity_labels: Dict[int, str] = {}
        self.model_format: Optional[str] = None
        self.model_path: Optional[Path] = None
        self.is_loaded = False

    def _load_activity_labels(self) -> None:
        if not settings.ACTIVITY_LABELS_PATH.exists():
            logger.warning(f"活动标签文件不存在：{settings.ACTIVITY_LABELS_PATH}")
            return

        activity_labels: Dict[int, str] = {}
        for line in settings.ACTIVITY_LABELS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            label_id_str, label_name = line.split(maxsplit=1)
            activity_labels[int(label_id_str)] = label_name

        self.activity_labels = activity_labels

    def _to_activity_name(self, raw_label: Any) -> str:
        if isinstance(raw_label, np.generic):
            raw_label = raw_label.item()

        if isinstance(raw_label, (int, np.integer)):
            label_id = int(raw_label)
            return self.activity_labels.get(label_id, f"UNKNOWN_{label_id}")

        text = str(raw_label)
        if text.isdigit():
            label_id = int(text)
            return self.activity_labels.get(label_id, text)

        return text

    def load(self):
        candidates = [
            settings.MODEL_PATH,
            MODEL_DIR / "svm_baseline.joblib",
            MODEL_DIR / "har_model.json",
        ]
        model_path = next((path for path in candidates if path.exists()), None)
        if model_path is None:
            logger.warning("模型文件不存在，传感器预测功能将不可用")
            return
        self.model_path = model_path

        try:
            self._load_activity_labels()
            if settings.FEATURES_PATH.exists():
                self.feature_names = joblib.load(str(settings.FEATURES_PATH))

            if model_path.suffix.lower() == ".json":
                from xgboost import XGBClassifier

                if not settings.ENCODER_PATH.exists():
                    raise FileNotFoundError(f"缺少标签编码器文件：{settings.ENCODER_PATH}")

                self.model = XGBClassifier()
                self.model.load_model(str(model_path))
                self.label_encoder = joblib.load(str(settings.ENCODER_PATH))
                self.model_format = "xgboost"
            else:
                self.model = joblib.load(str(model_path))
                self.label_encoder = None
                self.model_format = "joblib"

            self.is_loaded = True
            logger.info(f"Model loaded: {model_path} ({self.model_format})")
        except Exception as e:
            logger.error(f"Model load failed: {e}", exc_info=True)

    def predict(self, features: list[float]) -> str:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未就绪")
        
        try:
            features_array = np.array(features).reshape(1, -1)
            logger.debug(f"输入特征维度：{features_array.shape}")
            
            raw_prediction = self.model.predict(features_array)[0]
            logger.debug(f"预测原始值：{raw_prediction}, 类型：{type(raw_prediction)}")

            if self.model_format == "xgboost" and self.label_encoder is not None:
                if hasattr(self.label_encoder, "inverse_transform"):
                    activity = self.label_encoder.inverse_transform([raw_prediction])[0]
                elif hasattr(self.label_encoder, "classes_"):
                    if 0 <= int(raw_prediction) < len(self.label_encoder.classes_):
                        activity = self.label_encoder.classes_[int(raw_prediction)]
                    else:
                        activity = f"UNKNOWN_{int(raw_prediction)}"
                else:
                    activity = str(raw_prediction)
            else:
                activity = raw_prediction
            
            activity = self._to_activity_name(activity)
            
            logger.debug(f"预测活动：{activity}, 类型：{type(activity)}")
            return activity
            
        except Exception as e:
            logger.error(f"模型预测异常：{e}", exc_info=True)
            raise RuntimeError(f"模型预测失败：{e}") 

model_manager = ModelManager()

PROMPT_SUSPECT_FALL = "检测到用户可能摔倒，请用一句简短的话询问用户是否安全，语气要温和。"
PROMPT_CONFIRMED_FALL = "用户已确认摔倒且长时间未起身，请生成一句紧急求助信息，包含时间和地点（假设地点为家中），用于通知紧急联系人。信息要简洁。"

async def process_activity_logic(user_id: str, activity: str, current_time: datetime):
    state = await state_manager.get_or_create(user_id)
    
    if activity == state.last_activity:
        state.activity_count += 1
    else:
        state.activity_count = 1
        state.last_activity = activity

    if state.activity_count >= settings.ACTION_THRESHOLD:
        effective_activity = activity
        fall_related = ["LAYING", "SITTING"]
        normal_activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "STANDING"]

        if state.current_state == "normal":
            if effective_activity in fall_related and state.last_effective_activity in normal_activities:
                state.current_state = "suspect_fall"
                state.suspect_start_time = current_time
                logger.info(f"[{current_time}] 用户 {user_id} 进入【疑似摔倒】状态")
                
                ask_phrase = await llm_service.generate(PROMPT_SUSPECT_FALL)
                state.last_question = ask_phrase if ask_phrase else "您还好吗？"

        elif state.current_state == "suspect_fall":
            if state.suspect_start_time:
                time_elapsed = (current_time - state.suspect_start_time).total_seconds()
                if effective_activity in fall_related and time_elapsed > settings.SUSPECT_TIMEOUT:
                    state.current_state = "confirmed_fall"
                    logger.info(f"[{current_time}] 用户 {user_id} 进入【确认摔倒】状态（持续 {time_elapsed:.0f} 秒）")
                    
                    emergency_msg = await llm_service.generate(PROMPT_CONFIRMED_FALL)
                    state.last_emergency_msg = emergency_msg if emergency_msg else "用户可能摔倒，请立即救援！"
                
                elif effective_activity not in fall_related:
                    state.current_state = "normal"
                    state.suspect_start_time = None
                    logger.info(f"[{current_time}] 用户 {user_id} 恢复【正常】状态")

        elif state.current_state == "confirmed_fall":
            if effective_activity not in fall_related:
                state.current_state = "normal"
                state.suspect_start_time = None
                logger.info(f"[{current_time}] 用户 {user_id} 从确认摔倒恢复【正常】状态")

        state.last_effective_activity = effective_activity

    logger.debug(f"用户 {user_id} | 动作：{activity} | 计数：{state.activity_count} | 状态：{state.current_state}")
    return state.current_state

async def build_fall_reaction(user_id: str, predicted_activity: str) -> str:
    state = await state_manager.get_or_create(user_id)
    if state.current_state == "confirmed_fall":
        return state.last_emergency_msg or "检测到较高跌倒风险，请立刻确认用户安全并考虑通知紧急联系人。"
    if state.current_state == "suspect_fall":
        return state.last_question or "我检测到你可能跌倒了，你现在还好吗？"
    if predicted_activity in {"LAYING", "SITTING"}:
        return f"当前识别到的动作是 {predicted_activity}，系统暂时保持观察，没有形成持续跌倒判定。"
    return f"当前识别到的动作是 {predicted_activity}，状态保持正常。"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在启动服务...")
    model_manager.load()
    yield
    logger.info("正在关闭服务...")
    await llm_service.close()
    await close_wellness_service()

app = FastAPI(title="摔倒检测后端", description="集成 XGBoost 与千问大模型", lifespan=lifespan)

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

@app.get("/")
async def read_root(request: Request):
    accept = request.headers.get("accept", "")
    if UI_DIR.exists() and "text/html" in accept:
        return FileResponse(UI_DIR / "index.html")
    return {
        "message": "摔倒检测服务运行中",
        "status": "running",
        "model_loaded": model_manager.is_loaded,
        "llm_configured": bool(settings.QWEN_API_KEY and settings.QWEN_API_KEY != "your-api-key-here")
    }

@app.get("/app")
async def read_app():
    if not UI_DIR.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(UI_DIR / "index.html")

@app.post("/wellness/report")
async def generate_wellness_report(payload: WellnessReportRequest):
    try:
        return await create_wellness_report(payload)
    except Exception as e:
        logger.error(f"Wellness report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"健康报告生成失败: {e}")

@app.post("/wellness/chat")
async def wellness_chat(payload: WellnessChatRequest):
    result = await chat_with_session(payload)
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在，请先生成报告")
    return result

@app.get("/wellness/demo-user/{user_id}")
async def get_demo_user(user_id: str):
    profile = load_demo_user(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="样例用户不存在")
    return profile

@app.get("/wellness/medical-record/{session_id}")
async def download_medical_record(session_id: str):
    built = await build_formal_record_file(session_id)
    if built is None:
        raise HTTPException(status_code=404, detail="未找到对应的正式记录会话")
    file_path, file_name = built
    return FileResponse(path=file_path, media_type="text/markdown; charset=utf-8", filename=file_name)

@app.post("/wellness/fall-detection")
async def wellness_fall_detection(payload: FallDetectionRequest):
    motion_data = payload.motion_data or {}
    timestamp = motion_data.get("timestamp")
    event_time = datetime.now()
    if isinstance(timestamp, str):
        try:
            event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            event_time = datetime.now()

    predicted_activity = motion_data.get("activity_label")
    if predicted_activity:
        predicted_activity = str(predicted_activity)
    else:
        features = motion_data.get("features")
        if not isinstance(features, list) or len(features) != 561:
            raise HTTPException(status_code=400, detail="请提供 activity_label 或长度为 561 的 features")
        if not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="动作模型尚未加载完成")
        try:
            predicted_activity = model_manager.predict([float(value) for value in features])
        except Exception as e:
            logger.error(f"Fall detection prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"动作识别失败: {e}")

    user_state = await process_activity_logic(payload.user_id, str(predicted_activity), event_time)
    state_snapshot = await state_manager.get_status(payload.user_id)
    reaction = await build_fall_reaction(payload.user_id, str(predicted_activity))
    return {
        "status": "ok",
        "predicted_activity": str(predicted_activity),
        "user_state": str(user_state),
        "reaction": reaction,
        "state_snapshot": state_snapshot,
    }

@app.get("/test-api")
async def test_api_connection():
    result = await llm_service.generate("你好，请简单回复。")
    return {"status": "success" if result else "failed", "response": result}

@app.get("/status/{user_id}")
async def get_user_status(user_id: str):
    status_data = await state_manager.get_status(user_id)
    if not status_data:
        return {"message": "用户不存在", "default_state": "normal"}
    return status_data

@app.post("/predict")
async def receive_prediction(pred: ActionPrediction):
    current_state = await process_activity_logic(pred.user_id, pred.activity_label, pred.timestamp)
    return {
        "status": "ok",
        "user_state": str(current_state),
        "received": str(pred.activity_label)
    }

@app.post("/sensor_predict")
async def sensor_predict(sensor_data: SensorData):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        logger.info(f"收到传感器数据：用户={sensor_data.user_id}, 特征维度={len(sensor_data.features)}")
        activity = model_manager.predict(sensor_data.features)
        logger.info(f"预测结果：{activity}")
    except Exception as e:
        logger.error(f"预测失败：{e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"模型预测错误：{str(e)}"
        )
    
    current_state = await process_activity_logic(sensor_data.user_id, activity, sensor_data.timestamp)
    
    return {
        "status": "ok",
        "predicted_activity": str(activity),
        "user_state": str(current_state)
    }

@app.post("/user_reply")
async def handle_user_reply(reply: UserReply):
    state_obj = await state_manager.get_or_create(reply.user_id)
    
    if state_obj.current_state not in ["suspect_fall", "confirmed_fall"]:
        return {"status": "ignored", "message": "当前状态不需要处理回复"}

    prompt = f"""
    用户回复了："{reply.reply_text}"
    请判断用户当前的状态是需要救援还是安全了。
    如果你认为用户需要紧急救援，请回复数字 2；
    如果你认为用户已经安全或只是短暂不适，请回复数字 1。
    只回复数字，不要有其他文字。
    """
    
    intent_raw = await llm_service.generate(prompt, system_message="你是一个摔倒检测系统的意图识别模块。")
    intent_code = None
    
    if intent_raw:
        match = re.search(r'[12]', intent_raw)
        if match:
            intent_code = match.group()
    
    if intent_code == "2":
        if state_obj.current_state != "confirmed_fall":
            await state_manager.update_state(reply.user_id, "confirmed_fall", reset_suspect_time=True)
            logger.info(f"[{reply.timestamp}] 用户 {reply.user_id} 根据回复确认为【确认摔倒】")
    elif intent_code == "1":
        await state_manager.update_state(reply.user_id, "normal", reset_suspect_time=True)
        logger.info(f"[{reply.timestamp}] 用户 {reply.user_id} 根据回复恢复【正常】状态")
    else:
        logger.info(f"意图识别失败或未确定，保持原状态：{state_obj.current_state}")

    updated_state = await state_manager.get_or_create(reply.user_id)

    return {
        "status": "processed",
        "intent": intent_code,
        "current_state": str(updated_state.current_state)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower())
