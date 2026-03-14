from __future__ import annotations

import json

import requests

from project_config import LEGACY_PORT, OUTPUT_DIR, UCI_HAR_ROOT


BASE_URL = f"http://127.0.0.1:{LEGACY_PORT}"
RESULT_PATH = OUTPUT_DIR / "test_with_real_data_result.json"
X_TEST_PATH = UCI_HAR_ROOT / "test" / "X_test.txt"


def _load_feature_row() -> list[float]:
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {X_TEST_PATH}")
    with X_TEST_PATH.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise RuntimeError("X_test.txt is empty")
    features = [float(item) for item in first_line.split()]
    if len(features) != 561:
        raise RuntimeError(f"Unexpected feature size: {len(features)}")
    return features


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": "real_data_test_user",
        "features": _load_feature_row(),
    }
    response = requests.post(f"{BASE_URL}/sensor_predict", json=payload, timeout=25)
    response.raise_for_status()
    result = {
        "base_url": BASE_URL,
        "payload_feature_dim": len(payload["features"]),
        "response": response.json(),
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "result_path": str(RESULT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
