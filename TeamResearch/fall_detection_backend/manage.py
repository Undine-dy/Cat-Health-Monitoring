from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import uvicorn

from project_config import FUSION_HOST, FUSION_PORT, FUSION_STRESS_DEMO_PORT, LEGACY_HOST, LEGACY_PORT


BASE_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> int:
    completed = subprocess.run(args, cwd=str(BASE_DIR))
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "serve-legacy",
            "serve-fusion",
            "serve-fusion-stress-port",
            "train-fusion",
            "train-stress",
            "demo-fusion",
            "demo-stress",
            "test-api",
            "test-real",
        ],
    )
    args = parser.parse_args()

    if args.task == "serve-legacy":
        uvicorn.run("main:app", host=LEGACY_HOST, port=LEGACY_PORT, log_level="info")
        return 0

    if args.task == "serve-fusion":
        uvicorn.run("fusion_main:app", host=FUSION_HOST, port=FUSION_PORT, log_level="info")
        return 0

    if args.task == "serve-fusion-stress-port":
        uvicorn.run("fusion_main:app", host=FUSION_HOST, port=FUSION_STRESS_DEMO_PORT, log_level="info")
        return 0

    if args.task == "train-fusion":
        return _run([sys.executable, "train_fusion_models.py"])

    if args.task == "train-stress":
        return _run([sys.executable, "train_stress_models.py"])

    if args.task == "demo-fusion":
        return _run([sys.executable, "run_fusion_demo.py"])

    if args.task == "demo-stress":
        return _run([sys.executable, "run_stress_demo.py"])

    if args.task == "test-api":
        return _run([sys.executable, "test_api.py"])

    if args.task == "test-real":
        return _run([sys.executable, "test_with_real_data.py"])

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
