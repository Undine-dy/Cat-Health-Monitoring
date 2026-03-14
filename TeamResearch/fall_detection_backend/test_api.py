from __future__ import annotations

import json
from datetime import datetime

import requests

from project_config import LEGACY_PORT, OUTPUT_DIR


BASE_URL = f"http://127.0.0.1:{LEGACY_PORT}"
RESULT_PATH = OUTPUT_DIR / "test_api_result.json"


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    response = requests.request(method=method, url=f"{BASE_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    user_id = "api_test_user"
    now = datetime.now().isoformat()

    root = _request("GET", "/")
    predict = _request(
        "POST",
        "/predict",
        {
            "user_id": user_id,
            "activity_label": "LAYING",
            "timestamp": now,
        },
    )
    status = _request("GET", f"/status/{user_id}")

    result = {
        "base_url": BASE_URL,
        "root": root,
        "predict": predict,
        "status": status,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "result_path": str(RESULT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
