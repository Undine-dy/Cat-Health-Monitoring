from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fusion_engine import FusionEngine
from fusion_schemas import SensorObservation
from project_config import FUSION_MODEL_DIR, OUTPUT_DIR


load_dotenv(Path(__file__).resolve().parent / ".env")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)

ACTIVITY_MODEL_PATH = FUSION_MODEL_DIR / "activity_context_model.joblib"
HR_REFERENCE_PATH = FUSION_MODEL_DIR / "hr_reference.json"
STRESS_MODEL_PATH = FUSION_MODEL_DIR / "stress_classifier.joblib"
VIRTUAL_DATA_PATH = Path(__file__).resolve().parent.parent / "New_dataset" / "virtual_people_report_input.json"


class WellnessReportRequest(BaseModel):
    user_id: str = Field(default="cat_user")
    data: Any


class WellnessChatRequest(BaseModel):
    session_id: str
    question: str


@dataclass
class WellnessSession:
    session_id: str
    user_id: str
    created_at: str
    source_type: str
    summary: dict[str, Any]
    report: str
    response: str
    formal_record: str
    chat_history: list[dict[str, str]] = field(default_factory=list)
    record_path: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, WellnessSession] = {}
        self._lock = asyncio.Lock()

    async def save(self, session: WellnessSession) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session

    async def get(self, session_id: str) -> WellnessSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def add_chat(self, session_id: str, role: str, content: str) -> WellnessSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.chat_history.append({"role": role, "content": content})
            return session

    async def set_record_path(self, session_id: str, record_path: str) -> WellnessSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.record_path = record_path
            return session


class QwenClient:
    def __init__(self) -> None:
        self.api_key = QWEN_API_KEY
        self.model = QWEN_MODEL
        self.api_url = QWEN_API_URL
        self.client = httpx.AsyncClient(timeout=90.0)

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your-api-key-here")

    async def close(self) -> None:
        await self.client.aclose()

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.4, max_tokens: int = 1200) -> str | None:
        if not self.configured:
            return None

        api_url = self.api_url
        if "chat/completions" not in api_url:
            api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.85,
        }

        try:
            response = await self.client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception:
            return None

        if "choices" in body and body["choices"]:
            text = body["choices"][0].get("message", {}).get("content", "")
            return text.strip() or None
        if "output" in body and isinstance(body["output"], dict):
            text = body["output"].get("text", "")
            return text.strip() or None
        return None


session_store = SessionStore()
qwen_client = QwenClient()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _extract_xyz(record: dict[str, Any], vector_key: str, x_key: str, y_key: str, z_key: str, default: list[float]) -> list[float]:
    direct = record.get(vector_key)
    if isinstance(direct, list) and len(direct) >= 3:
        return [_safe_float(direct[0]), _safe_float(direct[1]), _safe_float(direct[2])]
    return [
        _safe_float(record.get(x_key), default[0]),
        _safe_float(record.get(y_key), default[1]),
        _safe_float(record.get(z_key), default[2]),
    ]


def _iter_leaf_values(obj: Any, prefix: str = "") -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    numeric: dict[str, list[float]] = defaultdict(list)
    textual: dict[str, list[str]] = defaultdict(list)

    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                next_path = f"{path}.{key}" if path else str(key)
                walk(value, next_path)
            return
        if isinstance(current, list):
            for index, value in enumerate(current):
                next_path = f"{path}[{index}]"
                walk(value, next_path)
            return
        if _is_number(current):
            numeric[path.lower()].append(float(current))
            return
        if isinstance(current, str):
            text = current.strip()
            if text and len(text) <= 200:
                textual[path.lower()].append(text)

    walk(obj, prefix)
    return numeric, textual


def _collect_matches(source: dict[str, list[Any]], aliases: list[str]) -> list[Any]:
    matches: list[Any] = []
    for path, values in source.items():
        if any(alias in path for alias in aliases):
            matches.extend(values)
    return matches


def _normalize_score(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 1.0:
        return max(0.0, value)
    if value <= 10.0:
        return max(0.0, min(1.0, value / 10.0))
    if value <= 100.0:
        return max(0.0, min(1.0, value / 100.0))
    return None


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "未提供"
    return f"{value:.2f}"


def _format_number(value: float | None, unit: str = "") -> str:
    if value is None:
        return "未提供"
    number = float(value)
    if abs(number) >= 100 or number.is_integer():
        return f"{int(round(number))}{unit}"
    return f"{number:.1f}{unit}"


def _summarize_distribution(dist: dict[str, int]) -> str:
    if not dist:
        return "暂无"
    ranked = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    return "，".join(f"{name} {count}" for name, count in ranked[:4])


def _suggestions_from_flags(flags: list[str], source_type: str) -> list[str]:
    suggestions: list[str] = []
    if any("睡眠" in flag for flag in flags):
        suggestions.append("先把睡眠恢复放到优先级最高的位置，今晚尽量争取完整的 7.5 到 8 小时休息。")
    if any("压力" in flag for flag in flags):
        suggestions.append("把高压任务拆成更短的节奏，中间安排 5 到 10 分钟离屏缓冲，避免一直绷着。")
    if any("久坐" in flag for flag in flags):
        suggestions.append("每 45 到 60 分钟起身活动 3 到 5 分钟，先把久坐打断。")
    if any("心率" in flag for flag in flags):
        suggestions.append("这几天尽量减少额外刺激，先观察静息状态下的心率是否能回落。")
    if any("恢复" in flag or "疲劳" in flag for flag in flags):
        suggestions.append("不要在恢复分偏低时继续叠加强度，先补休息和补水，再考虑训练或高强度工作。")
    if not suggestions:
        suggestions.append("继续保持规律作息、适度活动和稳定补水，优先关注连续几天的变化。")
    if source_type == "fusion_timeline" and len(suggestions) < 3:
        suggestions.append("如果后续仍持续触发预警，建议把同一时间段的数据再复测一次，比较趋势是否缓解。")
    return suggestions[:4]


def _top_concern(summary: dict[str, Any]) -> str:
    flags = summary.get("flags", [])
    if not flags:
        return "整体状态暂时平稳"
    return flags[0]


def _build_caring_response(summary: dict[str, Any]) -> str:
    source_type = summary.get("source_type")
    metrics = summary.get("metrics", {})
    flags = summary.get("flags", [])

    if source_type == "fusion_timeline":
        sleep_debt = metrics.get("sleep_debt_hours")
        stress_avg = metrics.get("stress_score_avg")
        recovery_avg = metrics.get("recovery_score_avg")
        alert_count = metrics.get("alert_count")
        if sleep_debt is not None and sleep_debt >= 1.5:
            return f"看起来你最近的恢复明显不够，最近 3 天累计睡眠债大约有 {sleep_debt:.1f} 小时。最近是不是入睡晚、容易醒，或者一直没有完整休息？"
        if stress_avg is not None and stress_avg >= 0.68:
            return f"这组数据里的压力负荷偏高，平均压力分接近 {_format_ratio(stress_avg)}。最近是不是一直在赶任务、情绪很紧，或者身体总觉得放不下来？"
        if recovery_avg is not None and recovery_avg < 0.38:
            return f"你的恢复分偏低，现在更像是在硬撑而不是恢复。最近有没有持续疲惫、注意力下降，或者运动后恢复特别慢的情况？"
        if alert_count:
            return f"这段监测里已经出现了 {int(alert_count)} 次预警，我建议先把身体负荷降下来。最近有没有哪段时间特别不舒服？"

    sleep_hours = metrics.get("sleep_hours")
    stress_score = metrics.get("stress_score")
    sedentary_minutes = metrics.get("sedentary_minutes")
    resting_hr = metrics.get("resting_heart_rate")
    if sleep_hours is not None and sleep_hours < 6.5:
        return f"看起来你最近睡得不太够，平均睡眠大约只有 {sleep_hours:.1f} 小时。最近是加班、熬夜，还是入睡本身就不太顺？"
    if stress_score is not None and stress_score >= 0.68:
        return f"你的压力指标偏高，而且有点像持续紧绷的状态。最近是不是有一件事一直压在心里，或者节奏根本停不下来？"
    if sedentary_minutes is not None and sedentary_minutes >= 180:
        return f"你这段时间久坐得比较久，身体恢复也可能被拖住了。最近是不是连续坐着学习或工作，几乎没怎么起身活动？"
    if resting_hr is not None and resting_hr >= 90:
        return f"你的静息心率偏高一些，身体可能还在持续应激。最近有没有心慌、休息后也放松不下来的感觉？"
    if flags:
        return f"我先注意到的是“{flags[0]}”。如果你愿意，可以告诉我最近最困扰你的那个点，我可以顺着继续帮你看。"
    return "这组数据里暂时没有特别尖锐的异常，但我还是建议你关注接下来几天有没有连续变化。你最近最想先改善哪一块？"


def _build_generic_summary(user_id: str, payload: Any) -> dict[str, Any]:
    numeric, textual = _iter_leaf_values(payload)
    sleep_values = [value for value in _collect_matches(numeric, ["sleep_hours", "sleep", "睡眠", "night_sleep", "睡眠时长"]) if 0 <= value <= 24]
    steps_values = [value for value in _collect_matches(numeric, ["steps", "step_count", "步数"]) if value >= 0]
    exercise_values = [value for value in _collect_matches(numeric, ["exercise_minutes", "active_minutes", "运动", "activity_minutes", "训练"]) if 0 <= value <= 1440]
    sedentary_values = [value for value in _collect_matches(numeric, ["sedentary", "久坐", "sitting", "inactive_minutes", "screen_time"]) if 0 <= value <= 1440]
    water_values = [value for value in _collect_matches(numeric, ["water", "hydration", "饮水", "water_ml"]) if 0 <= value <= 6000]
    resting_hr_values = [value for value in _collect_matches(numeric, ["resting_heart_rate", "resting_hr", "静息心率"]) if 20 <= value <= 240]
    stress_values = [_normalize_score(value) for value in _collect_matches(numeric, ["stress_score", "stress", "压力", "anxiety", "焦虑"])]
    stress_values = [value for value in stress_values if value is not None]
    mood_values = [_normalize_score(value) for value in _collect_matches(numeric, ["mood", "情绪"])]
    mood_values = [value for value in mood_values if value is not None]

    note_values = _collect_matches(textual, ["notes", "note", "remark", "症状", "symptom", "complaint", "主诉", "情况"])
    notes = "；".join(dict.fromkeys(note_values))[:300]

    sleep_hours = _mean(sleep_values)
    steps = _mean(steps_values)
    exercise_minutes = _mean(exercise_values)
    sedentary_minutes = _mean(sedentary_values)
    water_ml = _mean(water_values)
    resting_heart_rate = _mean(resting_hr_values)
    stress_score = _mean(stress_values)
    mood_score = _mean(mood_values)

    flags: list[str] = []
    if sleep_hours is not None and sleep_hours < 6.5:
        flags.append(f"睡眠时长偏少，平均约 {sleep_hours:.1f} 小时")
    if stress_score is not None and stress_score >= 0.68:
        flags.append(f"压力评分偏高，约为 {stress_score:.2f}")
    if sedentary_minutes is not None and sedentary_minutes >= 180:
        flags.append(f"久坐时间偏长，约 {sedentary_minutes:.0f} 分钟")
    if exercise_minutes is not None and exercise_minutes < 20:
        flags.append(f"运动时间偏少，约 {exercise_minutes:.0f} 分钟")
    if water_ml is not None and water_ml < 1200:
        flags.append(f"饮水量偏低，约 {water_ml:.0f} ml")
    if resting_heart_rate is not None and resting_heart_rate >= 90:
        flags.append(f"静息心率偏高，约 {resting_heart_rate:.0f} 次/分")
    if mood_score is not None and mood_score <= 0.35:
        flags.append(f"情绪状态偏低，约 {mood_score:.2f}")
    if not flags:
        flags.append("当前上传数据没有明显高风险信号")

    suggestions = _suggestions_from_flags(flags, "generic_json")
    overview = []
    if sleep_hours is not None:
        overview.append(f"睡眠 {_format_number(sleep_hours, ' 小时')}")
    if stress_score is not None:
        overview.append(f"压力 {_format_ratio(stress_score)}")
    if exercise_minutes is not None:
        overview.append(f"运动 {_format_number(exercise_minutes, ' 分钟')}")
    if sedentary_minutes is not None:
        overview.append(f"久坐 {_format_number(sedentary_minutes, ' 分钟')}")
    if resting_heart_rate is not None:
        overview.append(f"静息心率 {_format_number(resting_heart_rate, ' 次/分')}")

    metrics = {
        "sleep_hours": sleep_hours,
        "steps": steps,
        "exercise_minutes": exercise_minutes,
        "sedentary_minutes": sedentary_minutes,
        "water_ml": water_ml,
        "resting_heart_rate": resting_heart_rate,
        "stress_score": stress_score,
        "mood_score": mood_score,
    }
    report_context = {
        "user_id": user_id,
        "source_type": "generic_json",
        "overview_metrics": metrics,
        "flags": flags,
        "notes": notes,
    }
    return {
        "user_id": user_id,
        "source_type": "generic_json",
        "overview_line": "，".join(overview) if overview else "已接收一组通用健康数据",
        "flags": flags,
        "suggestions": suggestions,
        "notes": notes,
        "metrics": metrics,
        "report_context": report_context,
    }


def _build_fusion_summary(user_id: str, payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("timeline"), list):
        records = payload["timeline"]
    elif isinstance(payload, list):
        records = payload
    else:
        return None

    if not records or not isinstance(records[0], dict):
        return None

    required_a = {"heart_rate", "wrist_acc_x", "wrist_acc_y", "wrist_acc_z", "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"}
    required_b = {"heart_rate", "wrist_acc", "wrist_gyro"}
    sample_keys = set(records[0].keys())
    if not required_a.issubset(sample_keys) and not required_b.issubset(sample_keys):
        return None

    engine = FusionEngine(
        activity_model_path=ACTIVITY_MODEL_PATH,
        hr_reference_path=HR_REFERENCE_PATH,
        stress_model_path=STRESS_MODEL_PATH if STRESS_MODEL_PATH.exists() else None,
    )

    all_results: list[dict[str, Any]] = []
    base_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    previous_time: datetime | None = None

    for record in records:
        record_time = _parse_iso_datetime(record.get("timestamp"))
        if record_time is None:
            sequence_second = record.get("sequence_second")
            if _is_number(sequence_second):
                record_time = base_time + timedelta(seconds=float(sequence_second))
            elif previous_time is not None:
                record_time = previous_time + timedelta(minutes=1)
            else:
                record_time = base_time

        observation = SensorObservation(
            user_id=_normalize_text(record.get("user_id") or record.get("person_id") or user_id) or user_id,
            timestamp=record_time,
            heart_rate=_safe_float(record.get("heart_rate"), 70.0),
            wrist_acc=_extract_xyz(record, "wrist_acc", "wrist_acc_x", "wrist_acc_y", "wrist_acc_z", [0.0, 0.0, 9.8]),
            wrist_gyro=_extract_xyz(record, "wrist_gyro", "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z", [0.0, 0.0, 0.0]),
            wrist_bvp=record.get("wrist_bvp"),
            wrist_eda=record.get("wrist_eda"),
            wrist_temp=record.get("wrist_temp"),
            context_override=record.get("activity_context") or record.get("context_override"),
        )
        previous_time = record_time
        all_results.append(engine.ingest(observation))

    if not all_results:
        return None

    stress_scores = [result["stress"]["stress_score"] for result in all_results]
    fatigue_scores = [result["fatigue"]["score"] for result in all_results]
    recovery_scores = [result["recovery"]["score"] for result in all_results]
    hr_levels: dict[str, int] = defaultdict(int)
    stress_levels: dict[str, int] = defaultdict(int)
    contexts: dict[str, int] = defaultdict(int)
    alerts = engine.get_alerts(user_id)

    for result in all_results:
        hr_levels[result["heart_rate"]["level"]] += 1
        stress_levels[result["stress"]["stress_level"]] += 1
        contexts[result["context"]["smoothed"]] += 1

    final_state = dict(all_results[-1])
    final_state.pop("explanation", None)
    habits = final_state["stress"]["habits"]

    metrics = {
        "record_count": len(all_results),
        "start_time": all_results[0]["timestamp"],
        "end_time": all_results[-1]["timestamp"],
        "stress_score_avg": round(float(np.mean(stress_scores)), 4),
        "fatigue_score_avg": round(float(np.mean(fatigue_scores)), 4),
        "recovery_score_avg": round(float(np.mean(recovery_scores)), 4),
        "alert_count": len(alerts),
        "heart_rate_distribution": dict(hr_levels),
        "stress_distribution": dict(stress_levels),
        "context_distribution": dict(contexts),
        "sleep_debt_hours": habits.get("sleep_debt_hours_3d"),
        "avg_sleep_hours": habits.get("avg_sleep_hours_3d"),
        "sedentary_minutes": habits.get("sedentary_minutes_today"),
        "active_minutes": habits.get("active_minutes_today"),
        "current_heart_rate": final_state["heart_rate"]["current"],
        "expected_heart_rate": final_state["heart_rate"]["expected"],
        "heart_rate_level": final_state["heart_rate"]["level"],
        "stress_level": final_state["stress"]["stress_level"],
        "fatigue_level": final_state["fatigue"]["level"],
        "recovery_level": final_state["recovery"]["level"],
        "reason_codes": final_state.get("reason_codes", []),
    }

    flags: list[str] = []
    if metrics["sleep_debt_hours"] is not None and metrics["sleep_debt_hours"] >= 1.5:
        flags.append(f"睡眠债明显，最近 3 天约 {metrics['sleep_debt_hours']:.1f} 小时")
    if metrics["stress_score_avg"] >= 0.68:
        flags.append(f"平均压力偏高，约 {metrics['stress_score_avg']:.2f}")
    if metrics["recovery_score_avg"] < 0.38:
        flags.append(f"恢复分偏低，约 {metrics['recovery_score_avg']:.2f}")
    if metrics["fatigue_score_avg"] >= 0.66:
        flags.append(f"疲劳分偏高，约 {metrics['fatigue_score_avg']:.2f}")
    if metrics["sedentary_minutes"] is not None and metrics["sedentary_minutes"] >= 180:
        flags.append(f"久坐时间较长，约 {metrics['sedentary_minutes']:.0f} 分钟")
    if metrics["alert_count"] > 0:
        flags.append(f"监测期间触发 {int(metrics['alert_count'])} 次预警")
    if metrics["heart_rate_level"] in {"warning", "critical"}:
        flags.append(
            f"当前心率相对上下文基线偏高，当前 {metrics['current_heart_rate']:.0f} 次/分，参考 {metrics['expected_heart_rate']:.0f} 次/分"
        )
    if not flags:
        flags.append("监测周期内整体状态相对平稳")

    suggestions = _suggestions_from_flags(flags, "fusion_timeline")
    report_context = {
        "user_id": user_id,
        "source_type": "fusion_timeline",
        "monitoring_window": {
            "start": metrics["start_time"],
            "end": metrics["end_time"],
            "record_count": metrics["record_count"],
        },
        "aggregated_metrics": {
            "stress_score_avg": metrics["stress_score_avg"],
            "fatigue_score_avg": metrics["fatigue_score_avg"],
            "recovery_score_avg": metrics["recovery_score_avg"],
            "alert_count": metrics["alert_count"],
            "heart_rate_distribution": metrics["heart_rate_distribution"],
            "stress_distribution": metrics["stress_distribution"],
            "context_distribution": metrics["context_distribution"],
            "sleep_debt_hours": metrics["sleep_debt_hours"],
            "avg_sleep_hours": metrics["avg_sleep_hours"],
            "sedentary_minutes": metrics["sedentary_minutes"],
            "active_minutes": metrics["active_minutes"],
            "current_heart_rate": metrics["current_heart_rate"],
            "expected_heart_rate": metrics["expected_heart_rate"],
            "heart_rate_level": metrics["heart_rate_level"],
            "stress_level": metrics["stress_level"],
            "fatigue_level": metrics["fatigue_level"],
            "recovery_level": metrics["recovery_level"],
        },
        "reason_codes": metrics["reason_codes"],
        "flags": flags,
    }
    overview_line = (
        f"监测共 {metrics['record_count']} 条记录，场景分布为 {_summarize_distribution(metrics['context_distribution'])}，"
        f"平均压力 {metrics['stress_score_avg']:.2f}，平均恢复 {metrics['recovery_score_avg']:.2f}"
    )
    return {
        "user_id": user_id,
        "source_type": "fusion_timeline",
        "overview_line": overview_line,
        "flags": flags,
        "suggestions": suggestions,
        "notes": "",
        "metrics": metrics,
        "final_state": final_state,
        "report_context": report_context,
    }


def summarize_user_data(user_id: str, payload: Any) -> dict[str, Any]:
    fusion_summary = _build_fusion_summary(user_id, payload)
    if fusion_summary is not None:
        return fusion_summary
    return _build_generic_summary(user_id, payload)


def _fallback_report(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    flags = summary.get("flags", [])
    suggestions = summary.get("suggestions", [])
    source_type = summary.get("source_type", "generic_json")

    lines = [
        "## 总览",
        summary.get("overview_line", "已完成这组数据的健康生活分析。"),
        "",
        "## 重点发现",
    ]
    for index, flag in enumerate(flags[:5], start=1):
        lines.append(f"{index}. {flag}")

    lines.extend(["", "## 生活方式提醒"])
    if source_type == "fusion_timeline":
        lines.append(
            f"当前心率约 {_format_number(metrics.get('current_heart_rate'), ' 次/分')}，"
            f"相对当前情境基线约 {_format_number(metrics.get('expected_heart_rate'), ' 次/分')}。"
        )
        lines.append(
            f"睡眠债约 {_format_number(metrics.get('sleep_debt_hours'), ' 小时')}，"
            f"活动分布为 {_summarize_distribution(metrics.get('context_distribution', {}))}。"
        )
    else:
        lines.append(
            f"睡眠 {_format_number(metrics.get('sleep_hours'), ' 小时')}，"
            f"压力 {_format_ratio(metrics.get('stress_score'))}，"
            f"久坐 {_format_number(metrics.get('sedentary_minutes'), ' 分钟')}。"
        )
        if summary.get("notes"):
            lines.append(f"用户补充信息：{summary['notes']}")

    lines.extend(["", "## 建议"])
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(f"{index}. {suggestion}")

    lines.extend(
        [
            "",
            "## 提醒",
            "这份结果基于你上传的数据做智能分析，只能作为健康管理参考。如果有明显不适或症状持续加重，还是要尽快线下就医。",
        ]
    )
    return "\n".join(lines)


async def _generate_report(summary: dict[str, Any]) -> str:
    context_json = json.dumps(summary.get("report_context", {}), ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个谨慎、具体、有人情味的健康生活助手。"
                "请基于给定的结构化健康摘要，用中文输出一份简洁但信息充分的报告。"
                "不要编造未提供的数据，不要下确定性医学诊断。"
                "请使用以下结构：## 总览、## 重点指标、## 生活方式提醒、## 建议、## 提醒。"
            ),
        },
        {
            "role": "user",
            "content": f"请分析这组健康摘要并生成报告：\n```json\n{context_json}\n```",
        },
    ]
    result = await qwen_client.chat(messages, temperature=0.35, max_tokens=1400)
    return result or _fallback_report(summary)


def _fallback_chat_answer(summary: dict[str, Any], question: str) -> str:
    metrics = summary.get("metrics", {})
    question_text = question.strip()
    if "睡" in question_text:
        if summary.get("source_type") == "fusion_timeline":
            debt = metrics.get("sleep_debt_hours")
            avg_sleep = metrics.get("avg_sleep_hours")
            return (
                f"从这组数据看，最近的恢复问题和睡眠关系很大。"
                f"最近 3 天的睡眠债大约是 {_format_number(debt, ' 小时')}，平均睡眠约 {_format_number(avg_sleep, ' 小时')}。"
                "如果你最近也主观觉得累，我会建议先把连续几晚的完整睡眠补回来，再观察压力和恢复分有没有回升。"
            )
        sleep_hours = metrics.get("sleep_hours")
        return (
            f"这组数据里你的平均睡眠大约是 {_format_number(sleep_hours, ' 小时')}。"
            "如果你最近还有白天犯困、醒后不清爽或入睡困难，这基本就值得优先处理。"
        )
    if "压力" in question_text or "紧张" in question_text or "焦虑" in question_text:
        stress_value = metrics.get("stress_score_avg")
        if stress_value is None:
            stress_value = metrics.get("stress_score")
        return (
            f"我更担心的是持续性的高压负荷，当前相关指标大约是 {_format_ratio(stress_value)}。"
            "如果这种状态已经连续几天没有缓下来，单靠硬扛通常只会继续消耗恢复能力。"
        )
    if "运动" in question_text or "久坐" in question_text or "活动" in question_text:
        if summary.get("source_type") == "fusion_timeline":
            active = metrics.get("active_minutes")
            sedentary = metrics.get("sedentary_minutes")
            return (
                f"从监测结果看，活跃时间大约 {_format_number(active, ' 分钟')}，久坐时间约 {_format_number(sedentary, ' 分钟')}。"
                "关键不是马上加很多运动量，而是先把长时间不动打断，让身体有机会从应激里退下来。"
            )
        exercise = metrics.get("exercise_minutes")
        sedentary = metrics.get("sedentary_minutes")
        return (
            f"你当前的数据更像是活动不足和久坐偏多并存，运动约 {_format_number(exercise, ' 分钟')}，久坐约 {_format_number(sedentary, ' 分钟')}。"
            "如果你愿意，我可以继续帮你把它拆成更容易执行的一天安排。"
        )
    if "心率" in question_text:
        current_hr = metrics.get("current_heart_rate")
        expected_hr = metrics.get("expected_heart_rate")
        resting_hr = metrics.get("resting_heart_rate")
        if current_hr is not None and expected_hr is not None:
            return (
                f"当前监测心率大约 {_format_number(current_hr, ' 次/分')}，对应情境参考值约 {_format_number(expected_hr, ' 次/分')}。"
                "如果你是在静息时也偏高，就更值得结合最近睡眠、压力和主观不适一起看。"
            )
        return (
            f"你上传的数据里静息心率大约 {_format_number(resting_hr, ' 次/分')}。"
            "如果最近伴随心慌、胸闷或明显头晕，就不要只看数字，最好及时线下评估。"
        )
    return (
        f"如果只看这份数据，我现在最先想追问的是“{_top_concern(summary)}”。"
        "你可以直接告诉我最近最明显的不舒服、作息变化，或者你最想先改善的那一项，我会顺着这个点继续拆给你。"
    )


async def _chat_answer(session: WellnessSession, question: str) -> str:
    trimmed_history = session.chat_history[-8:]
    context_json = json.dumps(session.summary.get("report_context", {}), ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个具体、温和、克制的健康陪伴助手。"
                "请根据已有健康摘要和报告回答用户追问。"
                "回答必须具体且人性化，2 到 5 句话，不要做确定性诊断。"
                "能引用具体数字时就引用，超出数据范围时要明确说不知道。"
            ),
        },
        {
            "role": "user",
            "content": f"这是已有健康摘要：\n```json\n{context_json}\n```\n\n这是已有报告：\n{session.report}",
        },
    ]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})
    result = await qwen_client.chat(messages, temperature=0.45, max_tokens=700)
    return result or _fallback_chat_answer(session.summary, question)


def _fallback_formal_record(summary: dict[str, Any], report: str) -> str:
    now = datetime.now()
    record_id = f"CAT-{now:%Y%m%d-%H%M%S}"
    flags = summary.get("flags", [])
    suggestions = summary.get("suggestions", [])
    metrics = summary.get("metrics", {})
    lines = [
        "# 医学诊断正式记录",
        "",
        f"记录编号：{record_id}",
        f"监测对象：{summary.get('user_id', '未知用户')}",
        f"生成时间：{now:%Y-%m-%d %H:%M:%S}",
        f"数据来源：{'融合监测数据' if summary.get('source_type') == 'fusion_timeline' else '用户提交健康数据'}",
        "",
        "## 一、资料摘要",
        summary.get("overview_line", "已完成健康状态分析。"),
        "",
        "## 二、主要发现",
    ]
    for index, flag in enumerate(flags[:5], start=1):
        lines.append(f"{index}. {flag}")
    lines.extend(["", "## 三、临床印象"])
    if summary.get("source_type") == "fusion_timeline":
        lines.append(
            f"综合监测结果提示当前以压力负荷、疲劳累积和恢复不足为主要关注方向。"
            f"压力均值约 {_format_ratio(metrics.get('stress_score_avg'))}，恢复均值约 {_format_ratio(metrics.get('recovery_score_avg'))}。"
        )
    else:
        lines.append(
            "结合用户提交的作息和生活方式数据，当前更需要优先处理的是影响恢复的生活习惯问题，"
            "并持续观察睡眠、压力和活动量是否出现连续改善。"
        )
    lines.extend(["", "## 四、建议与随访"])
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(f"{index}. {suggestion}")
    lines.extend(
        [
            "5. 建议在 3 到 7 天内复测同类数据，重点比较睡眠、压力、恢复和主观不适是否同步改善。",
            "",
            "## 五、记录说明",
            "本正式记录仅保留临床化整理后的结论与建议，不包含融合系统原始指标数据、原始数组或模型概率细节。",
            "",
            "## 六、免责声明",
            "本记录由健康监测数据自动整理生成，仅供健康管理参考，不能替代线下医生面诊与正式医学诊断。如有持续不适，请及时就医。",
            "",
            "## 附加摘要",
            report,
        ]
    )
    return "\n".join(lines)


async def _generate_formal_record(summary: dict[str, Any], report: str) -> str:
    context_json = json.dumps(summary.get("report_context", {}), ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "你是医疗文书助手。请把给定健康摘要整理成中文正式记录。"
                "必须包含：一、基本信息；二、主要发现；三、临床印象；四、建议与随访；五、免责声明。"
                "可以引用少量关键数字，但不要包含原始 JSON、原始指标数组、融合系统原始指标数据、模型概率或 reason_codes。"
                "语气要专业、客观、克制。"
            ),
        },
        {
            "role": "user",
            "content": f"请整理正式记录。\n健康摘要：\n```json\n{context_json}\n```\n\n已有分析报告：\n{report}",
        },
    ]
    result = await qwen_client.chat(messages, temperature=0.25, max_tokens=1800)
    return result or _fallback_formal_record(summary, report)


async def create_wellness_report(payload: WellnessReportRequest) -> dict[str, Any]:
    summary = summarize_user_data(payload.user_id, payload.data)
    report = await _generate_report(summary)
    response = _build_caring_response(summary)
    formal_record = await _generate_formal_record(summary, report)
    session_id = uuid.uuid4().hex[:12]
    session = WellnessSession(
        session_id=session_id,
        user_id=summary["user_id"],
        created_at=datetime.now().isoformat(),
        source_type=summary["source_type"],
        summary=summary,
        report=report,
        response=response,
        formal_record=formal_record,
        chat_history=[{"role": "assistant", "content": response}],
    )
    await session_store.save(session)
    return {
        "session_id": session_id,
        "user_id": summary["user_id"],
        "source_type": summary["source_type"],
        "overview": summary.get("overview_line"),
        "flags": summary.get("flags", []),
        "report": report,
        "response": response,
        "formal_record_ready": True,
    }


async def chat_with_session(payload: WellnessChatRequest) -> dict[str, Any] | None:
    session = await session_store.get(payload.session_id)
    if session is None:
        return None
    await session_store.add_chat(payload.session_id, "user", payload.question)
    answer = await _chat_answer(session, payload.question)
    await session_store.add_chat(payload.session_id, "assistant", answer)
    return {
        "session_id": payload.session_id,
        "answer": answer,
        "user_id": session.user_id,
    }


async def build_formal_record_file(session_id: str) -> tuple[Path, str] | None:
    session = await session_store.get(session_id)
    if session is None:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = Path(session.record_path) if session.record_path else None
    if existing_path and existing_path.exists():
        return existing_path, existing_path.name
    file_name = f"official_medical_record_{session.user_id}_{session.session_id}.md"
    output_path = OUTPUT_DIR / file_name
    output_path.write_text(session.formal_record, encoding="utf-8")
    await session_store.set_record_path(session_id, str(output_path))
    return output_path, file_name


def load_demo_user(user_id: str) -> dict[str, Any] | None:
    if not VIRTUAL_DATA_PATH.exists():
        return None
    profiles = json.loads(VIRTUAL_DATA_PATH.read_text(encoding="utf-8"))
    for profile in profiles:
        if profile.get("person_id") == user_id:
            return profile
    return None


async def close_wellness_service() -> None:
    await qwen_client.close()
