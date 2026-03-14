import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX2,FMA3,AVX512F")

import numpy as np
import requests

from project_config import BACKEND_DIR, FUSION_HOST, FUSION_STRESS_DEMO_PORT, OUTPUT_DIR, WESAD_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


BASE_DIR = BACKEND_DIR
DATASET_ROOT = WESAD_ROOT
CLIENT_HOST = "127.0.0.1" if FUSION_HOST in {"0.0.0.0", "::"} else FUSION_HOST
BASE_URL = f"http://{CLIENT_HOST}:{FUSION_STRESS_DEMO_PORT}"


def wait_for_server() -> None:
    for _ in range(50):
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("fusion backend did not start")


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == target_len:
        return arr.astype(np.float32, copy=False)
    indices = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.int64)
    return arr[indices].astype(np.float32, copy=False)


def _load_resampled_subject(subject_id: str) -> dict:
    pkl_path = DATASET_ROOT / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing subject file: {pkl_path}")
    with pkl_path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")

    wrist = payload["signal"]["wrist"]
    labels_raw = np.asarray(payload["label"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1, 1).astype(np.float32, copy=False)
    target_len = int(eda.shape[0])
    acc = _resample_to_length(np.asarray(wrist["ACC"]), target_len)
    bvp = _resample_to_length(np.asarray(wrist["BVP"]), target_len).reshape(-1)
    temp = _resample_to_length(np.asarray(wrist["TEMP"]), target_len).reshape(-1)
    labels = _resample_to_length(labels_raw, target_len).reshape(-1).astype(np.int32, copy=False)
    return {"acc": acc, "bvp": bvp, "eda": eda.reshape(-1), "temp": temp, "labels": labels}


def _find_segment(labels: np.ndarray, label_value: int, min_len: int) -> tuple[int, int]:
    indices = np.where(labels == label_value)[0]
    if indices.size < min_len:
        raise RuntimeError(f"Not enough samples for label={label_value}: {indices.size}")

    best_start, best_end = 0, 0
    run_start = int(indices[0])
    prev = int(indices[0])
    for current in indices[1:]:
        current = int(current)
        if current != prev + 1:
            if prev - run_start > best_end - best_start:
                best_start, best_end = run_start, prev
            run_start = current
        prev = current
    if prev - run_start > best_end - best_start:
        best_start, best_end = run_start, prev

    if best_end - best_start + 1 < min_len:
        start = int(indices[0])
        return start, start + min_len
    center = (best_start + best_end) // 2
    half = min_len // 2
    return center - half, center - half + min_len


def _norm3(arr: np.ndarray) -> float:
    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    return float(np.sqrt(x * x + y * y + z * z))


def _build_rows(subject: dict, label_value: int, duration_seconds: int = 90) -> list[dict]:
    sample_rate_hz = 4
    total = duration_seconds * sample_rate_hz
    start, end = _find_segment(subject["labels"], label_value=label_value, min_len=total)
    rows = []
    bvp_slice = subject["bvp"][start:end]
    bvp_mean = float(np.mean(bvp_slice))

    for i in range(total):
        idx = start + i
        acc = subject["acc"][idx]
        acc_mag = _norm3(acc)
        label_hr_base = 72.0 if label_value == 1 else 96.0 if label_value == 2 else 80.0
        hr = label_hr_base + (acc_mag - 9.8) * 1.8
        rows.append(
            {
                "heart_rate": float(max(45.0, min(180.0, hr))),
                "wrist_acc": [float(acc[0]), float(acc[1]), float(acc[2])],
                "wrist_gyro": [0.0, 0.0, 0.0],
                "wrist_bvp": float(subject["bvp"][idx] - bvp_mean),
                "wrist_eda": float(subject["eda"][idx]),
                "wrist_temp": float(subject["temp"][idx]),
            }
        )
    return rows


def post_observation(payload: dict) -> dict:
    response = requests.post(f"{BASE_URL}/fusion/ingest", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def checkpoint(user_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/fusion/state/{user_id}", timeout=20)
    response.raise_for_status()
    return response.json()


def _state_accuracy(pred_states: list[str], expected_state: str) -> float:
    if not pred_states:
        return 0.0
    return float(np.mean(np.asarray(pred_states) == expected_state))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_DIR / "stress_demo_server_stdout.log"
    stderr_path = OUTPUT_DIR / "stress_demo_server_stderr.log"

    subject = _load_resampled_subject("S3")
    baseline_rows = _build_rows(subject, label_value=1, duration_seconds=90)
    stress_rows = _build_rows(subject, label_value=2, duration_seconds=90)

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "fusion_main:app", "--host", FUSION_HOST, "--port", str(FUSION_STRESS_DEMO_PORT)],
            cwd=str(BASE_DIR),
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            wait_for_server()

            user_id = "stress_demo_user"
            current_time = datetime(2026, 3, 11, 10, 0, 0)
            baseline_scores: list[float] = []
            stress_scores: list[float] = []
            baseline_states: list[str] = []
            stress_states: list[str] = []
            baseline_raw_states: list[str] = []
            stress_raw_states: list[str] = []
            cold_start_count = 0
            primary_count = 0
            primary_padded_count = 0

            for row in baseline_rows:
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": row["heart_rate"],
                    "wrist_acc": row["wrist_acc"],
                    "wrist_gyro": row["wrist_gyro"],
                    "ppg_quality": 0.98,
                    "wrist_bvp": row["wrist_bvp"],
                    "wrist_eda": row["wrist_eda"],
                    "wrist_temp": row["wrist_temp"],
                    "context_override": "resting",
                }
                response = post_observation(payload)
                baseline_scores.append(float(response["stress"]["stress_score"]))
                model_info = response["stress"].get("model", {})
                if model_info.get("used"):
                    baseline_states.append(str(model_info.get("state")))
                    baseline_raw_states.append(str(model_info.get("raw_state", model_info.get("state"))))
                    if model_info.get("model_name") == "cold_start":
                        cold_start_count += 1
                    if model_info.get("model_name") == "primary":
                        primary_count += 1
                    if model_info.get("model_name") == "primary_padded":
                        primary_padded_count += 1
                current_time += timedelta(milliseconds=250)

            baseline_checkpoint = checkpoint(user_id)

            for row in stress_rows:
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": row["heart_rate"],
                    "wrist_acc": row["wrist_acc"],
                    "wrist_gyro": row["wrist_gyro"],
                    "ppg_quality": 0.98,
                    "wrist_bvp": row["wrist_bvp"],
                    "wrist_eda": row["wrist_eda"],
                    "wrist_temp": row["wrist_temp"],
                    "context_override": "resting",
                }
                response = post_observation(payload)
                stress_scores.append(float(response["stress"]["stress_score"]))
                model_info = response["stress"].get("model", {})
                if model_info.get("used"):
                    stress_states.append(str(model_info.get("state")))
                    stress_raw_states.append(str(model_info.get("raw_state", model_info.get("state"))))
                    if model_info.get("model_name") == "cold_start":
                        cold_start_count += 1
                    if model_info.get("model_name") == "primary":
                        primary_count += 1
                    if model_info.get("model_name") == "primary_padded":
                        primary_padded_count += 1
                current_time += timedelta(milliseconds=250)

            final_state = checkpoint(user_id)
            alerts = requests.get(f"{BASE_URL}/fusion/alerts/{user_id}", timeout=20).json()

            def top_state(items: list[str]) -> str:
                if not items:
                    return "n/a"
                values, counts = np.unique(np.asarray(items), return_counts=True)
                return str(values[int(np.argmax(counts))])

            baseline_acc = _state_accuracy(baseline_states, "baseline")
            stress_acc = _state_accuracy(stress_states, "stress")
            total_pred = len(baseline_states) + len(stress_states)
            total_correct = int(round(baseline_acc * len(baseline_states))) + int(round(stress_acc * len(stress_states)))
            overall_acc = float(total_correct / total_pred) if total_pred > 0 else 0.0

            result = {
                "baseline_checkpoint": baseline_checkpoint,
                "final_state": final_state,
                "alerts": alerts,
                "metrics": {
                    "baseline_avg_stress_score": round(float(np.mean(baseline_scores)), 6),
                    "stress_stage_avg_stress_score": round(float(np.mean(stress_scores)), 6),
                    "delta_stress_score": round(float(np.mean(stress_scores) - np.mean(baseline_scores)), 6),
                    "baseline_top_model_state": top_state(baseline_states),
                    "stress_stage_top_model_state": top_state(stress_states),
                    "baseline_state_accuracy": round(baseline_acc, 6),
                    "stress_state_accuracy": round(stress_acc, 6),
                    "overall_state_accuracy": round(overall_acc, 6),
                    "used_predictions": int(total_pred),
                    "cold_start_predictions": int(cold_start_count),
                    "primary_predictions": int(primary_count),
                    "primary_padded_predictions": int(primary_padded_count),
                },
            }
            (OUTPUT_DIR / "stress_demo_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            report = [
                "# 压力智能体端到端演示",
                "",
                f"- baseline 平均压力分: {result['metrics']['baseline_avg_stress_score']}",
                f"- stress 段平均压力分: {result['metrics']['stress_stage_avg_stress_score']}",
                f"- 压力分变化: {result['metrics']['delta_stress_score']}",
                f"- baseline 段状态准确率: {result['metrics']['baseline_state_accuracy']}",
                f"- stress 段状态准确率: {result['metrics']['stress_state_accuracy']}",
                f"- 整体状态准确率: {result['metrics']['overall_state_accuracy']}",
                f"- 冷启动预测次数: {result['metrics']['cold_start_predictions']}",
                f"- 主模型预测次数: {result['metrics']['primary_predictions']}",
                f"- 主模型补齐预测次数: {result['metrics']['primary_padded_predictions']}",
                f"- 最终解释: {final_state['explanation']}",
                f"- 报警数量: {len(alerts['alerts'])}",
            ]
            (OUTPUT_DIR / "stress_demo_report.md").write_text("\n".join(report), encoding="utf-8")

            print(json.dumps(result["metrics"], ensure_ascii=False))
            return 0
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
