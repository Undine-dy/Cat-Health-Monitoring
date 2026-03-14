"""
End-to-end pipeline: Fusion Engine → Qianwen LLM → Standard Medical Report

Usage:
    python generate_medical_report.py [--user-id ID_010]

Reads patient sensor data from New_dataset/, runs it through the local
FusionEngine (no server required), strips the engine-generated explanation,
and sends the structured metrics to Qianwen to produce a clinically-formatted
wearable health monitoring report.  The final output is a single file written
to the project output/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from fusion_engine import FusionEngine
from fusion_schemas import SensorObservation
from project_config import FUSION_MODEL_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACTIVITY_MODEL_PATH = FUSION_MODEL_DIR / "activity_context_model.joblib"
HR_REFERENCE_PATH = FUSION_MODEL_DIR / "hr_reference.json"
STRESS_MODEL_PATH = FUSION_MODEL_DIR / "stress_classifier.joblib"
VIRTUAL_DATA_PATH = Path(__file__).resolve().parent.parent / "New_dataset" / "virtual_people_report_input.json"

# ---------------------------------------------------------------------------
# Qianwen config (reuse .env already in the project)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("MedicalReport")

# =========================================================================
# 1. Run fusion engine locally
# =========================================================================


def run_fusion_pipeline(user_id: str) -> dict:
    """Replay all sensor records for *user_id* through the FusionEngine and
    return aggregated metrics (with all ``explanation`` fields removed)."""

    logger.info("Loading virtual patient data …")
    profiles = json.loads(VIRTUAL_DATA_PATH.read_text(encoding="utf-8"))
    profile = next((p for p in profiles if p["person_id"] == user_id), None)
    if profile is None:
        raise SystemExit(f"Patient {user_id} not found in {VIRTUAL_DATA_PATH}")

    records = profile["timeline"]
    logger.info("Building FusionEngine …")
    engine = FusionEngine(
        activity_model_path=ACTIVITY_MODEL_PATH,
        hr_reference_path=HR_REFERENCE_PATH,
        stress_model_path=STRESS_MODEL_PATH if STRESS_MODEL_PATH.exists() else None,
    )

    # -- replay every record -------------------------------------------------
    base_time = datetime(2026, 3, 12, 9, 0, 0)
    all_results: list[dict] = []
    for idx, rec in enumerate(records):
        ts = base_time + timedelta(minutes=idx)
        obs = SensorObservation(
            user_id=user_id,
            timestamp=ts,
            heart_rate=rec["heart_rate"],
            wrist_acc=[rec["wrist_acc_x"], rec["wrist_acc_y"], rec["wrist_acc_z"]],
            wrist_gyro=[rec["wrist_gyro_x"], rec["wrist_gyro_y"], rec["wrist_gyro_z"]],
            wrist_bvp=rec.get("wrist_bvp"),
            wrist_eda=rec.get("wrist_eda"),
            wrist_temp=rec.get("wrist_temp"),
        )
        result = engine.ingest(obs)
        all_results.append(result)

    logger.info("Replay complete – %d records ingested.", len(all_results))

    # -- aggregate -----------------------------------------------------------
    stress_scores = [r["stress"]["stress_score"] for r in all_results]
    fatigue_scores = [r["fatigue"]["score"] for r in all_results]
    recovery_scores = [r["recovery"]["score"] for r in all_results]

    hr_levels: dict[str, int] = defaultdict(int)
    stress_levels: dict[str, int] = defaultdict(int)
    context_dist: dict[str, int] = defaultdict(int)
    model_states: dict[str, int] = defaultdict(int)
    alert_types: dict[str, int] = defaultdict(int)

    for r in all_results:
        hr_levels[r["heart_rate"]["level"]] += 1
        stress_levels[r["stress"]["stress_level"]] += 1
        context_dist[r["context"]["smoothed"]] += 1
        if r["stress"]["model"].get("used"):
            model_states[r["stress"]["model"]["state"]] += 1

    alerts = engine.get_alerts(user_id)
    for a in alerts:
        alert_types[a["type"]] += 1

    # Collect checkpoints at 25 / 50 / 75 / 100 %
    checkpoints = {}
    for frac in (0.25, 0.50, 0.75, 1.0):
        idx = min(int(len(all_results) * frac) - 1, len(all_results) - 1)
        cp = dict(all_results[idx])
        cp.pop("explanation", None)
        checkpoints[str(frac)] = cp

    final_state = dict(all_results[-1])
    final_state.pop("explanation", None)

    import numpy as np

    aggregated = {
        "run_info": {
            "selected_user": user_id,
            "total_records": len(all_results),
            "start_timestamp": all_results[0]["timestamp"],
            "end_timestamp": all_results[-1]["timestamp"],
        },
        "aggregated_metrics": {
            "stress_score_avg": round(float(np.mean(stress_scores)), 6),
            "stress_score_max": round(float(np.max(stress_scores)), 4),
            "stress_score_min": round(float(np.min(stress_scores)), 4),
            "fatigue_score_avg": round(float(np.mean(fatigue_scores)), 6),
            "fatigue_score_max": round(float(np.max(fatigue_scores)), 4),
            "recovery_score_avg": round(float(np.mean(recovery_scores)), 6),
            "recovery_score_min": round(float(np.min(recovery_scores)), 4),
            "heart_rate_level_distribution": dict(hr_levels),
            "stress_level_distribution": dict(stress_levels),
            "context_smoothed_distribution": dict(context_dist),
            "stress_model_state_distribution": dict(model_states),
            "alert_count": len(alerts),
            "alert_type_distribution": dict(alert_types),
        },
        "final_state": final_state,
        "alerts": {"user_id": user_id, "alerts": alerts},
        "checkpoints": checkpoints,
    }
    return aggregated


# =========================================================================
# 2. Call Qianwen to generate a standard medical report
# =========================================================================

SYSTEM_PROMPT = """\
你是一位资深临床医学信息系统专家，同时精通可穿戴设备健康监测数据的临床解读。

你的任务是：根据下方提供的可穿戴设备融合监测系统输出的结构化数据，撰写一份 **符合医疗信息系统规范** 的中文健康监测报告。

## 报告格式要求（严格遵循）

报告必须包含以下章节，并按顺序排列：

### 一、报告基本信息
- 报告名称：可穿戴设备健康状态监测报告
- 报告编号（自动生成，格式：WH-YYYYMMDD-XXXX）
- 监测对象 ID
- 监测时间段（起止时间）
- 数据采样总量
- 报告生成时间
- 报告类型：连续监测-自动分析报告

### 二、监测概要
用 2-3 段话概括本次监测的总体发现，包括活动类型分布、整体健康风险水平、是否触发预警。

### 三、生命体征分析
#### 3.1 心率监测
- 当前心率 / 运动上下文期望心率 / 偏差
- 心率水平分布（正常/警告/危急占比）
- 心率信号质量
- 临床意义解读

#### 3.2 压力评估
- 综合压力评分（0-1 量表）：均值 / 峰值 / 最低值
- 压力等级分布（低/中/高 占比）
- 监督学习模型状态分布与置信度
- 压力来源因子分析（reason_codes 权重排序）
- 临床意义解读

#### 3.3 疲劳与恢复
- 疲劳评分（均值/峰值）及等级
- 恢复评分（均值/最低值）及等级
- 临床意义解读

### 四、行为与生活方式分析
- 活动上下文分布（骑行/跑步/静息等各占多少分钟）
- 睡眠债（3 日均值 vs 目标 7.5h）
- 久坐时长分析
- 活跃分钟数
- 生活方式风险评估

### 五、预警事件汇总
- 预警总数与类型分布
- 按时间线列出关键预警（最多列出 5 条代表性事件）
- 预警级别与临床紧迫程度评估

### 六、趋势分析（基于 25%/50%/75%/100% 检查点）
分析监测周期内各指标的变化趋势，识别恶化或改善模式。

### 七、临床印象与建议
#### 7.1 临床印象
基于以上数据给出综合临床印象（2-3 句话）。

#### 7.2 健康建议
按优先级列出 3-5 条具体、可操作的健康建议。

#### 7.3 随访建议
给出随访监测的建议周期与重点关注指标。

### 八、免责声明
标准医疗免责声明：本报告由可穿戴设备数据自动生成，仅供参考，不构成医学诊断。如有健康疑虑请及时就医。

---

## 写作要求
1. 使用规范的医学术语，但保持可读性
2. 所有数值必须引用原始数据中的具体数字，不可编造
3. 临床解读必须基于循证医学常识
4. 报告语气应专业、客观、严谨
5. 压力评分解读参考：<0.35 低压力, 0.35-0.68 中等压力, >0.68 高压力
6. 疲劳评分解读参考：<0.33 低疲劳, 0.33-0.66 中等疲劳, >0.66 高疲劳
7. 恢复评分解读参考：≥0.65 良好, 0.38-0.65 中等, <0.38 低恢复
"""


def call_qianwen(metrics_json: str) -> str:
    """Send structured fusion metrics to Qianwen and return the medical report."""

    if not QWEN_API_KEY:
        raise SystemExit("QWEN_API_KEY is not configured in .env")

    user_prompt = (
        "以下是可穿戴设备融合监测系统输出的患者结构化数据（已移除系统自动解释字段，"
        "请你基于原始数据独立撰写临床解读）：\n\n"
        f"```json\n{metrics_json}\n```\n\n"
        "请严格按照上述报告格式要求，输出完整的健康监测报告。"
    )

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9,
    }

    logger.info("Calling Qianwen (%s) …", QWEN_MODEL)

    # Use OpenAI-compatible endpoint
    api_url = QWEN_API_URL
    if "compatible-mode" not in api_url and "chat/completions" not in api_url:
        api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()

    # Parse response (OpenAI-compatible format)
    if "choices" in body and len(body["choices"]) > 0:
        text = body["choices"][0].get("message", {}).get("content", "")
    elif "output" in body and "text" in body["output"]:
        text = body["output"]["text"]
    else:
        raise RuntimeError(f"Unexpected Qianwen response format: {json.dumps(body, ensure_ascii=False)[:500]}")

    if not text.strip():
        raise RuntimeError("Qianwen returned empty response")

    logger.info("Qianwen response received (%d chars).", len(text))
    return text.strip()


# =========================================================================
# 3. Assemble & write output
# =========================================================================


def _strip_explanations(obj):
    """Recursively remove all 'explanation' keys from a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _strip_explanations(v) for k, v in obj.items() if k != "explanation"}
    if isinstance(obj, list):
        return [_strip_explanations(item) for item in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a standard medical report via Qianwen LLM")
    parser.add_argument("--user-id", default="ID_010", help="Patient ID to process (default: ID_010)")
    args = parser.parse_args()

    user_id = args.user_id

    # Step 1: Run fusion pipeline
    logger.info("=== Step 1: Running fusion pipeline for %s ===", user_id)
    metrics = run_fusion_pipeline(user_id)

    # Step 2: Strip all explanation fields
    metrics_clean = _strip_explanations(metrics)
    metrics_json = json.dumps(metrics_clean, ensure_ascii=False, indent=2)

    # Step 3: Call Qianwen
    logger.info("=== Step 2: Calling Qianwen for medical report generation ===")
    medical_report = call_qianwen(metrics_json)

    # Step 4: Assemble final output
    logger.info("=== Step 3: Writing output ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"medical_report_{user_id}.md"

    header = (
        f"# 可穿戴设备健康状态监测报告\n\n"
        f"> **监测对象**: {user_id}  \n"
        f"> **监测时段**: {metrics['run_info']['start_timestamp']} → {metrics['run_info']['end_timestamp']}  \n"
        f"> **数据采样量**: {metrics['run_info']['total_records']} 条  \n"
        f"> **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"> **报告生成方式**: FusionEngine 数据采集 → 通义千问大模型({QWEN_MODEL})临床解读  \n\n"
        f"---\n\n"
    )

    separator = (
        "\n\n---\n\n"
        "## 附录：融合系统原始指标数据\n\n"
        "<details>\n<summary>点击展开原始 JSON 数据</summary>\n\n"
        f"```json\n{metrics_json}\n```\n\n"
        "</details>\n"
    )

    final_content = header + medical_report + separator
    output_path.write_text(final_content, encoding="utf-8")

    logger.info("Report written to %s", output_path)
    print(f"\n{'=' * 60}")
    print(f"  Medical report generated successfully!")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
