import json
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from project_config import BACKEND_DIR, FUSION_HOST, FUSION_PORT, OUTPUT_DIR, PAMAP2_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


BASE_DIR = BACKEND_DIR
DATA_PATH = PAMAP2_ROOT / "Protocol" / "subject108.dat"
CLIENT_HOST = "127.0.0.1" if FUSION_HOST in {"0.0.0.0", "::"} else FUSION_HOST
BASE_URL = f"http://{CLIENT_HOST}:{FUSION_PORT}"

PAMAP2_COLUMNS = [
    "timestamp", "activity_id", "heart_rate", "hand_temp",
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "hand_acc6_x", "hand_acc6_y", "hand_acc6_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_orient_1", "hand_orient_2", "hand_orient_3", "hand_orient_4",
    "chest_temp", "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "chest_acc6_x", "chest_acc6_y", "chest_acc6_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_orient_1", "chest_orient_2", "chest_orient_3", "chest_orient_4",
    "ankle_temp", "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
    "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_orient_1", "ankle_orient_2", "ankle_orient_3", "ankle_orient_4",
]

ACTIVITY_CONTEXT = {
    1: "resting",
    2: "resting",
    3: "resting",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "walking",
    12: "stairs",
    13: "stairs",
    16: "household",
    17: "household",
}


def wait_for_server() -> None:
    for _ in range(40):
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("fusion backend did not start")


def load_rows(activity_id: int, limit: int, step: int = 50) -> list[dict]:
    frame = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=PAMAP2_COLUMNS, na_values="NaN")
    frame = frame[frame["activity_id"] == activity_id].dropna(
        subset=["heart_rate", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z"]
    )
    frame = frame.iloc[::step].head(limit).copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "heart_rate": float(row["heart_rate"]),
                "wrist_acc": [float(row["hand_acc6_x"]), float(row["hand_acc6_y"]), float(row["hand_acc6_z"])],
                "wrist_gyro": [float(row["hand_gyro_x"]), float(row["hand_gyro_y"]), float(row["hand_gyro_z"])],
            }
        )
    return rows


def load_template_row(activity_id: int) -> dict:
    rows = load_rows(activity_id=activity_id, limit=1, step=1)
    if not rows:
        raise RuntimeError(f"no rows available for activity_id={activity_id}")
    return rows[0]


def post_observation(payload: dict) -> dict:
    response = requests.post(f"{BASE_URL}/fusion/ingest", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def checkpoint(user_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/fusion/state/{user_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_DIR / "fusion_demo_server_stdout.log"
    stderr_path = OUTPUT_DIR / "fusion_demo_server_stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "fusion_main:app", "--host", FUSION_HOST, "--port", str(FUSION_PORT)],
            cwd=str(BASE_DIR),
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            wait_for_server()
            user_id = "fusion_demo_user"
            current_time = datetime(2026, 3, 10, 1, 0, 0)
            timeline = []
            sleep_template = load_template_row(activity_id=1)
            sedentary_template = load_template_row(activity_id=2)

            for _ in range(240):
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": sleep_template["heart_rate"],
                    "wrist_acc": sleep_template["wrist_acc"],
                    "wrist_gyro": sleep_template["wrist_gyro"],
                    "context_override": "resting",
                }
                timeline.append(post_observation(payload))
                current_time += timedelta(minutes=1)

            sleep_checkpoint = checkpoint(user_id)

            current_time = datetime(2026, 3, 10, 9, 0, 0)
            for _ in range(180):
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": sedentary_template["heart_rate"],
                    "wrist_acc": sedentary_template["wrist_acc"],
                    "wrist_gyro": sedentary_template["wrist_gyro"],
                    "context_override": "resting",
                }
                timeline.append(post_observation(payload))
                current_time += timedelta(minutes=1)

            sedentary_checkpoint = checkpoint(user_id)

            for row in load_rows(activity_id=4, limit=25):
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": row["heart_rate"],
                    "wrist_acc": row["wrist_acc"],
                    "wrist_gyro": row["wrist_gyro"],
                    "context_override": "walking",
                }
                timeline.append(post_observation(payload))
                current_time += timedelta(minutes=1)

            for _ in range(6):
                payload = {
                    "user_id": user_id,
                    "timestamp": current_time.isoformat(),
                    "heart_rate": sedentary_template["heart_rate"] + 35.0,
                    "wrist_acc": sedentary_template["wrist_acc"],
                    "wrist_gyro": sedentary_template["wrist_gyro"],
                    "context_override": "resting",
                }
                timeline.append(post_observation(payload))
                current_time += timedelta(minutes=1)

            final_state = checkpoint(user_id)
            alerts = requests.get(f"{BASE_URL}/fusion/alerts/{user_id}", timeout=30).json()

            result = {
                "sleep_checkpoint": sleep_checkpoint,
                "sedentary_checkpoint": sedentary_checkpoint,
                "final_state": final_state,
                "alerts": alerts,
            }

            (OUTPUT_DIR / "fusion_demo_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report = [
                "# 融合系统演示结果",
                "",
                f"- 睡眠后状态：压力 {sleep_checkpoint['stress']['level'] if 'level' in sleep_checkpoint.get('stress', {}) else sleep_checkpoint['stress']['stress_level']}",
                f"- 久坐后解释：{sedentary_checkpoint['explanation']}",
                f"- 最终解释：{final_state['explanation']}",
                f"- 最终心率状态：{final_state['heart_rate']['level']}",
                f"- 报警数量：{len(alerts['alerts'])}",
            ]
            (OUTPUT_DIR / "fusion_demo_report.md").write_text("\n".join(report), encoding="utf-8")
            print(json.dumps({"alerts": len(alerts["alerts"]), "stress": final_state["stress"]["stress_level"]}, ensure_ascii=False))
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
