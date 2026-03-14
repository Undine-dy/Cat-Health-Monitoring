import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from activity_context_features import extract_window_features
from fusion_schemas import SensorObservation
from stress_features import RUNTIME_FEATURE_COLUMNS, extract_runtime_stress_features


CONTEXTS = {"resting", "walking", "running", "cycling", "stairs", "household"}
ACTIVE_CONTEXTS = {"walking", "running", "cycling", "stairs", "household"}
WORK_HOURS = range(9, 19)

STRESS_STATE_SCORE = {
    "baseline": 0.18,
    "meditation": 0.20,
    "amusement": 0.36,
    "stress": 0.90,
}


@dataclass
class DailyStats:
    resting_seconds: float = 0.0
    active_seconds: float = 0.0
    sedentary_seconds: float = 0.0
    current_sedentary_bout_seconds: float = 0.0
    longest_sedentary_bout_seconds: float = 0.0
    night_rest_seconds: float = 0.0
    unexpected_hr_seconds: float = 0.0


@dataclass
class UserFusionState:
    user_id: str
    last_timestamp: datetime | None = None
    context_history: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    current_context: str = "resting"
    raw_observation_buffer: deque[list[float]] = field(default_factory=lambda: deque(maxlen=512))
    stress_signal_buffer: deque[list[float]] = field(default_factory=lambda: deque(maxlen=512))
    stress_interval_seconds: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    stress_state_history: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    alerts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    daily_stats: dict[str, DailyStats] = field(default_factory=dict)
    context_hr_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    latest_result: dict[str, Any] | None = None
    latest_reason_codes: list[dict[str, Any]] = field(default_factory=list)
    hr_high_streak_seconds: float = 0.0
    hr_low_streak_seconds: float = 0.0


def _norm3(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    return float(np.sqrt(x * x + y * y + z * z))


class FusionEngine:
    def __init__(self, activity_model_path: Path, hr_reference_path: Path, stress_model_path: Path | None = None):
        model_bundle = joblib.load(activity_model_path)
        self.activity_model = model_bundle["model"]
        self.model_type = model_bundle.get("model_type", "single_frame_classifier")
        self.window_size = int(model_bundle.get("window_size", 1))
        self.raw_feature_names = model_bundle.get(
            "raw_feature_names",
            ["heart_rate", "wrist_acc_x", "wrist_acc_y", "wrist_acc_z", "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"],
        )
        self.hr_reference = json.loads(hr_reference_path.read_text(encoding="utf-8"))
        self.states: dict[str, UserFusionState] = {}

        self.stress_model = None
        self.stress_cold_start_model = None
        self.stress_label_encoder = None
        self.stress_labels: list[str] = []
        self.stress_window_size = 120
        self.stress_cold_start_window_size = 48
        self.stress_input_columns = list(RUNTIME_FEATURE_COLUMNS)

        if stress_model_path is not None and stress_model_path.exists():
            stress_bundle = joblib.load(stress_model_path)
            self.stress_model = stress_bundle.get("model")
            self.stress_cold_start_model = stress_bundle.get("cold_start_model")
            self.stress_label_encoder = stress_bundle.get("label_encoder")
            self.stress_labels = list(stress_bundle.get("labels", []))
            self.stress_window_size = int(stress_bundle.get("window_size", self.stress_window_size))
            self.stress_cold_start_window_size = int(
                stress_bundle.get("cold_start_window_size", self.stress_cold_start_window_size)
            )
            self.stress_input_columns = list(stress_bundle.get("input_columns", self.stress_input_columns))

    def ingest(self, observation: SensorObservation) -> dict[str, Any]:
        state = self.states.setdefault(observation.user_id, UserFusionState(user_id=observation.user_id))
        raw_context = observation.context_override or self.predict_context(state, observation)
        smoothed_context = self._smooth_context(state, raw_context)
        delta_seconds = self._compute_delta_seconds(state, observation.timestamp)
        self._append_stress_signal(state, observation, delta_seconds)

        self._update_daily_stats(state, observation, smoothed_context, delta_seconds)
        hr_status = self._evaluate_heart_rate(state, observation, smoothed_context, delta_seconds)
        stress_status = self._evaluate_stress(state, observation, smoothed_context, hr_status)
        explanation = self._build_explanation(stress_status, hr_status)

        result = {
            "user_id": observation.user_id,
            "timestamp": observation.timestamp.isoformat(),
            "context": {"raw": raw_context, "smoothed": smoothed_context},
            "heart_rate": hr_status,
            "stress": stress_status,
            "fatigue": {"score": round(stress_status["fatigue_score"], 4), "level": stress_status["fatigue_level"]},
            "recovery": {"score": round(stress_status["recovery_score"], 4), "level": stress_status["recovery_level"]},
            "habits": stress_status["habits"],
            "reason_codes": stress_status["reason_codes"],
            "explanation": explanation,
            "recent_alerts": list(state.alerts)[-5:],
        }
        state.latest_reason_codes = stress_status["reason_codes"]
        state.latest_result = result
        state.last_timestamp = observation.timestamp
        return result

    def predict_context(self, state: UserFusionState, observation: SensorObservation) -> str:
        raw_vector = [
            float(observation.heart_rate),
            float(observation.wrist_acc[0]),
            float(observation.wrist_acc[1]),
            float(observation.wrist_acc[2]),
            float(observation.wrist_gyro[0]),
            float(observation.wrist_gyro[1]),
            float(observation.wrist_gyro[2]),
        ]
        state.raw_observation_buffer.append(raw_vector)

        if self.model_type == "windowed_classifier":
            if len(state.raw_observation_buffer) < self.window_size:
                return state.current_context or "resting"
            window = np.asarray(list(state.raw_observation_buffer)[-self.window_size :], dtype=np.float32)
            acc_mag = np.sqrt(np.sum(window[:, 1:4] * window[:, 1:4], axis=1, keepdims=True))
            gyro_mag = np.sqrt(np.sum(window[:, 4:7] * window[:, 4:7], axis=1, keepdims=True))
            full_window = np.hstack([window, acc_mag, gyro_mag])
            features = extract_window_features(full_window).reshape(1, -1)
            return str(self.activity_model.predict(features)[0])

        acc = observation.wrist_acc
        gyro = observation.wrist_gyro
        features = np.array(
            [[float(acc[0]), float(acc[1]), float(acc[2]), float(gyro[0]), float(gyro[1]), float(gyro[2]), _norm3(acc), _norm3(gyro)]]
        )
        return str(self.activity_model.predict(features)[0])

    def get_state(self, user_id: str) -> dict[str, Any] | None:
        state = self.states.get(user_id)
        if not state:
            return None
        return state.latest_result

    def get_alerts(self, user_id: str) -> list[dict[str, Any]]:
        state = self.states.get(user_id)
        if not state:
            return []
        return list(state.alerts)

    def _smooth_context(self, state: UserFusionState, context: str) -> str:
        if context not in CONTEXTS:
            context = "resting"
        state.context_history.append(context)
        counts: dict[str, int] = defaultdict(int)
        for item in state.context_history:
            counts[item] += 1
        smoothed = max(sorted(counts), key=lambda item: counts[item])
        state.current_context = smoothed
        return smoothed

    def _compute_delta_seconds(self, state: UserFusionState, timestamp: datetime) -> float:
        if state.last_timestamp is None:
            return 60.0
        delta = (timestamp - state.last_timestamp).total_seconds()
        if delta <= 0:
            return 1.0
        return float(min(delta, 300.0))

    def _stats_for(self, state: UserFusionState, bucket: str) -> DailyStats:
        if bucket not in state.daily_stats:
            state.daily_stats[bucket] = DailyStats()
        return state.daily_stats[bucket]

    def _sleep_bucket(self, timestamp: datetime) -> str:
        if timestamp.hour >= 20:
            return (timestamp.date() + timedelta(days=1)).isoformat()
        return timestamp.date().isoformat()

    def _is_night(self, timestamp: datetime) -> bool:
        return timestamp.hour >= 22 or timestamp.hour < 8

    def _update_daily_stats(
        self,
        state: UserFusionState,
        observation: SensorObservation,
        context: str,
        delta_seconds: float,
    ) -> None:
        day_bucket = observation.timestamp.date().isoformat()
        stats = self._stats_for(state, day_bucket)

        is_resting = context == "resting"
        if is_resting:
            stats.resting_seconds += delta_seconds
        else:
            stats.active_seconds += delta_seconds

        sedentary_like = is_resting and observation.timestamp.hour in WORK_HOURS and _norm3(observation.wrist_acc) < 12.5
        if sedentary_like:
            stats.sedentary_seconds += delta_seconds
            stats.current_sedentary_bout_seconds += delta_seconds
            stats.longest_sedentary_bout_seconds = max(stats.longest_sedentary_bout_seconds, stats.current_sedentary_bout_seconds)
        else:
            stats.current_sedentary_bout_seconds = 0.0

        if self._is_night(observation.timestamp) and is_resting:
            sleep_stats = self._stats_for(state, self._sleep_bucket(observation.timestamp))
            sleep_stats.night_rest_seconds += delta_seconds

    def _append_stress_signal(self, state: UserFusionState, observation: SensorObservation, delta_seconds: float) -> None:
        row = [
            float(observation.wrist_acc[0]),
            float(observation.wrist_acc[1]),
            float(observation.wrist_acc[2]),
            _norm3(observation.wrist_acc),
            float(observation.wrist_bvp) if observation.wrist_bvp is not None else float("nan"),
            float(observation.wrist_eda) if observation.wrist_eda is not None else float("nan"),
            float(observation.wrist_temp) if observation.wrist_temp is not None else float("nan"),
        ]
        state.stress_signal_buffer.append(row)
        if delta_seconds > 0:
            state.stress_interval_seconds.append(float(delta_seconds))

    def _stress_sample_rate(self, state: UserFusionState) -> float:
        if not state.stress_interval_seconds:
            return 4.0
        median_delta = float(np.median(np.asarray(state.stress_interval_seconds, dtype=np.float32)))
        if median_delta <= 0:
            return 4.0
        sample_rate = 1.0 / median_delta
        return float(max(0.2, min(16.0, sample_rate)))

    def _decode_stress_label(self, pred_index: int) -> str:
        if self.stress_label_encoder is not None:
            decoded = self.stress_label_encoder.inverse_transform([int(pred_index)])
            return str(decoded[0])
        if 0 <= pred_index < len(self.stress_labels):
            return str(self.stress_labels[pred_index])
        return str(pred_index)

    def _smooth_stress_state(self, state: UserFusionState, raw_state: str) -> str:
        state.stress_state_history.append(raw_state)
        counts: dict[str, int] = defaultdict(int)
        for item in state.stress_state_history:
            counts[item] += 1
        return max(sorted(counts), key=lambda item: counts[item])

    def _predict_with_stress_model(self, model: Any, window: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
        feature_vector = extract_runtime_stress_features(window, sample_rate_hz=sample_rate_hz).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(feature_vector))[0]
            pred_index = int(np.argmax(probabilities))
            raw_state = self._decode_stress_label(pred_index)
            confidence = float(np.max(probabilities))
            probs_named = {self._decode_stress_label(i): round(float(probabilities[i]), 6) for i in range(probabilities.shape[0])}
            return {"raw_state": raw_state, "confidence": confidence, "probabilities": probs_named}

        pred_index = int(model.predict(feature_vector)[0])
        raw_state = self._decode_stress_label(pred_index)
        return {"raw_state": raw_state, "confidence": None, "probabilities": {raw_state: 1.0}}

    def _stress_model_inference(self, state: UserFusionState) -> dict[str, Any]:
        if self.stress_model is None:
            return {"enabled": False, "used": False, "reason": "model_unavailable"}

        available = len(state.stress_signal_buffer)
        use_model = None
        use_window = 0
        model_name = "primary"
        padded_from = 0
        if available >= self.stress_window_size:
            use_model = self.stress_model
            use_window = self.stress_window_size
            model_name = "primary"
        elif self.stress_cold_start_model is not None and available >= self.stress_cold_start_window_size:
            use_model = self.stress_cold_start_model
            use_window = self.stress_cold_start_window_size
            model_name = "cold_start"
        elif available >= self.stress_cold_start_window_size:
            use_model = self.stress_model
            use_window = self.stress_window_size
            model_name = "primary_padded"
            padded_from = available
        else:
            return {
                "enabled": True,
                "used": False,
                "reason": "insufficient_window",
                "required_window_size": int(self.stress_cold_start_window_size if self.stress_cold_start_model is not None else self.stress_window_size),
                "available_window_size": int(available),
            }

        history = np.asarray(list(state.stress_signal_buffer), dtype=np.float32)
        if model_name == "primary_padded":
            short_window = history[-padded_from:]
            pad_count = max(0, use_window - short_window.shape[0])
            if pad_count > 0:
                pad_block = np.repeat(short_window[:1], repeats=pad_count, axis=0)
                window = np.vstack([pad_block, short_window]).astype(np.float32, copy=False)
            else:
                window = short_window[-use_window:]
        else:
            window = history[-use_window:]
        if window.shape[1] != len(self.stress_input_columns):
            return {
                "enabled": True,
                "used": False,
                "reason": "feature_shape_mismatch",
                "expected_columns": list(self.stress_input_columns),
                "actual_shape": list(window.shape),
            }

        required_indices = [4, 5, 6]
        availability = [float(np.mean(np.isfinite(window[:, idx]))) for idx in required_indices]
        if min(availability) < 0.7:
            return {
                "enabled": True,
                "used": False,
                "reason": "missing_required_signals",
                "signal_availability": {
                    "bvp_ratio": round(availability[0], 4),
                    "eda_ratio": round(availability[1], 4),
                    "temp_ratio": round(availability[2], 4),
                },
            }

        sample_rate_hz = self._stress_sample_rate(state)
        pred = self._predict_with_stress_model(use_model, window, sample_rate_hz=sample_rate_hz)
        raw_state = str(pred["raw_state"])
        smoothed_state = self._smooth_stress_state(state, raw_state)

        return {
            "enabled": True,
            "used": True,
            "state": smoothed_state,
            "raw_state": raw_state,
            "confidence": round(float(pred["confidence"]), 6) if pred["confidence"] is not None else None,
            "probabilities": pred["probabilities"],
            "sample_rate_hz": round(sample_rate_hz, 4),
            "model_name": model_name,
            "window_size": int(use_window),
            "padded_from": int(padded_from) if padded_from > 0 else None,
        }

    def _default_reference(self, context: str) -> dict[str, float]:
        return self.hr_reference.get(context, self.hr_reference.get("resting", {}))

    def _context_baseline(self, state: UserFusionState, context: str) -> float:
        baseline = state.context_hr_baseline.get(context)
        if baseline and baseline.get("count", 0) > 0:
            return baseline["ema"]
        return float(self._default_reference(context).get("median", 70.0))

    def _update_context_baseline(
        self,
        state: UserFusionState,
        context: str,
        heart_rate: float,
        ppg_quality: float,
        allow_update: bool,
    ) -> None:
        if ppg_quality < 0.5 or not allow_update:
            return
        baseline = state.context_hr_baseline.setdefault(context, {"ema": self._context_baseline(state, context), "count": 0.0})
        alpha = 0.03 if context == "resting" else 0.015
        baseline["ema"] = round(alpha * heart_rate + (1 - alpha) * baseline["ema"], 4)
        baseline["count"] += 1.0

    def _evaluate_heart_rate(
        self,
        state: UserFusionState,
        observation: SensorObservation,
        context: str,
        delta_seconds: float,
    ) -> dict[str, Any]:
        reference = self._default_reference(context)
        expected = self._context_baseline(state, context)
        mad = float(reference.get("mad", 5.0))
        warning_band = max(12.0, mad * 2.2)
        critical_band = max(25.0, mad * 3.5)
        deviation = observation.heart_rate - expected

        level = "normal"
        direction = "normal"
        if deviation >= critical_band and context not in {"running", "cycling"}:
            level = "critical"
            direction = "high"
        elif deviation >= warning_band:
            level = "warning"
            direction = "high"
        elif deviation <= -critical_band and not self._is_night(observation.timestamp):
            level = "critical"
            direction = "low"
        elif deviation <= -warning_band:
            level = "warning"
            direction = "low"

        if context in {"running", "cycling"} and deviation > 0 and deviation < critical_band + 10.0:
            level = "normal"
            direction = "normal"

        if level != "normal":
            day_stats = self._stats_for(state, observation.timestamp.date().isoformat())
            day_stats.unexpected_hr_seconds += delta_seconds

        if direction == "high":
            state.hr_high_streak_seconds += delta_seconds
            state.hr_low_streak_seconds = 0.0
        elif direction == "low":
            state.hr_low_streak_seconds += delta_seconds
            state.hr_high_streak_seconds = 0.0
        else:
            state.hr_high_streak_seconds = 0.0
            state.hr_low_streak_seconds = 0.0

        should_alert = level == "critical" or state.hr_high_streak_seconds >= 180.0 or state.hr_low_streak_seconds >= 180.0
        if should_alert:
            alert = {
                "timestamp": observation.timestamp.isoformat(),
                "type": "heart_rate",
                "level": level,
                "direction": direction,
                "context": context,
                "heart_rate": round(observation.heart_rate, 2),
                "expected": round(expected, 2),
                "deviation": round(deviation, 2),
            }
            if not state.alerts or state.alerts[-1] != alert:
                state.alerts.append(alert)

        self._update_context_baseline(
            state=state,
            context=context,
            heart_rate=observation.heart_rate,
            ppg_quality=observation.ppg_quality,
            allow_update=(level == "normal"),
        )

        return {
            "current": round(observation.heart_rate, 2),
            "context": context,
            "expected": round(expected, 2),
            "expected_range": {"low": round(expected - warning_band, 2), "high": round(expected + warning_band, 2)},
            "deviation": round(deviation, 2),
            "level": level,
            "signal_quality": round(observation.ppg_quality, 4),
            "high_streak_seconds": round(state.hr_high_streak_seconds, 2),
            "low_streak_seconds": round(state.hr_low_streak_seconds, 2),
        }

    def _recent_sleep_hours(self, state: UserFusionState, current_date: date) -> list[float]:
        values = []
        for offset in range(0, 3):
            key = (current_date - timedelta(days=offset)).isoformat()
            stats = state.daily_stats.get(key)
            if stats:
                values.append(stats.night_rest_seconds / 3600.0)
        return values

    def _evaluate_stress(
        self,
        state: UserFusionState,
        observation: SensorObservation,
        context: str,
        hr_status: dict[str, Any],
    ) -> dict[str, Any]:
        current_day = observation.timestamp.date().isoformat()
        day_stats = self._stats_for(state, current_day)
        sleep_hours = self._recent_sleep_hours(state, observation.timestamp.date())
        sleep_target = float(self.hr_reference.get("defaults", {}).get("sleep_target_hours", 7.5))
        avg_sleep_hours = float(np.mean(sleep_hours)) if sleep_hours else 6.0
        sleep_debt_hours = max(0.0, sleep_target - avg_sleep_hours)
        current_sedentary_minutes = day_stats.current_sedentary_bout_seconds / 60.0
        longest_sedentary_minutes = day_stats.longest_sedentary_bout_seconds / 60.0
        hr_deviation = max(0.0, float(hr_status["deviation"]))

        contributions = {
            "sleep_debt": min(1.0, sleep_debt_hours / 3.0) * 0.38,
            "sedentary_bout": min(1.0, longest_sedentary_minutes / 150.0) * 0.24,
            "resting_hr_elevated": min(1.0, hr_deviation / 20.0) * (0.24 if context == "resting" else 0.12),
            "night_recovery_low": min(1.0, max(0.0, 5.5 - avg_sleep_hours) / 2.0) * 0.14,
        }
        heuristic_stress_score = min(1.0, sum(contributions.values()))

        model_status = self._stress_model_inference(state)
        stress_score = heuristic_stress_score
        model_weight = 0.0
        if model_status.get("used"):
            model_weight = 0.68
            weighted = 0.0
            for label, prob in model_status.get("probabilities", {}).items():
                weighted += STRESS_STATE_SCORE.get(str(label), 0.5) * float(prob)
            model_score = min(1.0, max(0.0, weighted))
            stress_score = min(1.0, max(0.0, model_weight * model_score + (1.0 - model_weight) * heuristic_stress_score))

            if str(model_status.get("state")) == "stress" and float(model_status.get("confidence") or 0.0) >= 0.82:
                alert = {
                    "timestamp": observation.timestamp.isoformat(),
                    "type": "stress",
                    "level": "high",
                    "state": "stress",
                    "confidence": round(float(model_status["confidence"]), 4),
                }
                if not state.alerts or state.alerts[-1] != alert:
                    state.alerts.append(alert)

        active_minutes_today = day_stats.active_seconds / 60.0
        fatigue_score = min(
            1.0,
            sleep_debt_hours / 4.0 * 0.40
            + min(1.0, active_minutes_today / 180.0) * 0.15
            + min(1.0, day_stats.unexpected_hr_seconds / 900.0) * 0.18
            + stress_score * 0.27,
        )
        recovery_score = max(0.0, 1.0 - (0.55 * stress_score + 0.45 * fatigue_score))

        reason_codes = [
            {"code": "sleep_debt_hours", "value": round(sleep_debt_hours, 2), "weight": round(contributions["sleep_debt"], 4)},
            {
                "code": "longest_sedentary_bout_minutes",
                "value": round(longest_sedentary_minutes, 2),
                "weight": round(contributions["sedentary_bout"], 4),
            },
            {
                "code": "heart_rate_above_context_baseline",
                "value": round(hr_deviation, 2),
                "weight": round(contributions["resting_hr_elevated"], 4),
            },
        ]
        if model_status.get("used"):
            reason_codes.append({"code": "stress_model_state", "value": str(model_status.get("state")), "weight": round(model_weight, 4)})
        elif model_status.get("reason"):
            reason_codes.append({"code": "stress_model_fallback_reason", "value": str(model_status.get("reason")), "weight": 0.0})
        reason_codes.sort(key=lambda item: float(item["weight"]), reverse=True)

        return {
            "stress_score": round(stress_score, 4),
            "stress_level": self._score_level(stress_score, low=0.35, high=0.68),
            "fatigue_score": round(fatigue_score, 4),
            "fatigue_level": self._score_level(fatigue_score, low=0.33, high=0.66),
            "recovery_score": round(recovery_score, 4),
            "recovery_level": "good" if recovery_score >= 0.65 else "moderate" if recovery_score >= 0.38 else "low",
            "habits": {
                "sleep_debt_hours_3d": round(sleep_debt_hours, 2),
                "avg_sleep_hours_3d": round(avg_sleep_hours, 2),
                "sedentary_minutes_today": round(day_stats.sedentary_seconds / 60.0, 2),
                "current_sedentary_bout_minutes": round(current_sedentary_minutes, 2),
                "longest_sedentary_bout_minutes": round(longest_sedentary_minutes, 2),
                "active_minutes_today": round(active_minutes_today, 2),
            },
            "reason_codes": reason_codes,
            "model": model_status,
        }

    def _score_level(self, score: float, low: float, high: float) -> str:
        if score >= high:
            return "high"
        if score >= low:
            return "moderate"
        return "low"

    def _build_explanation(self, stress_status: dict[str, Any], hr_status: dict[str, Any]) -> str:
        habits = stress_status["habits"]
        model_info = stress_status.get("model", {})
        fragments: list[str] = []

        if model_info.get("used") and model_info.get("state") == "stress":
            confidence = model_info.get("confidence")
            if confidence is None:
                fragments.append("监督模型判断当前更接近压力状态")
            else:
                fragments.append(f"监督模型判断当前更接近压力状态(置信度 {float(confidence):.2f})")
        elif model_info.get("used") and model_info.get("state") == "meditation":
            fragments.append("监督模型识别到更接近放松/恢复状态")

        if habits["sleep_debt_hours_3d"] >= 1.5:
            fragments.append(f"最近 3 天平均睡眠不足，睡眠债约 {habits['sleep_debt_hours_3d']:.1f} 小时")
        if habits["longest_sedentary_bout_minutes"] >= 90:
            fragments.append(f"今天出现约 {habits['longest_sedentary_bout_minutes']:.0f} 分钟连续久坐")
        if hr_status["deviation"] >= 12 and hr_status["context"] == "resting":
            fragments.append("静息状态下心率明显高于个人上下文基线")
        if not fragments:
            fragments.append("当前未见明显压力累积信号")

        return "；".join(fragments) + "。"
