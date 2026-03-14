# Wearable Fusion Backend

当前版本包含：

- 活动上下文识别（PAMAP2）
- 心率异常检测与报警
- 压力/疲劳/恢复状态评估
- WESAD 监督压力分类（XGBoost）
- 冷启动短窗策略（窗口不足时补齐到主窗口后再做监督推理）

## 主要脚本

- `train_fusion_models.py`：训练活动模型
- `train_stress_models.py`：训练压力模型
- `fusion_main.py`：API 服务入口
- `fusion_engine.py`：融合引擎
- `stress_features.py`：压力特征提取
- `run_fusion_demo.py`：融合流程演示
- `run_stress_demo.py`：压力流程演示

## 数据路径

- PAMAP2：`E:\moon\TeamResearch\Dataset\PAMAP2\PAMAP2_Dataset\PAMAP2_Dataset`
- WESAD：`E:\moon\TeamResearch\Dataset\WESAD_Kaggle\WESAD`

## 训练命令

```powershell
cd E:\moon\TeamResearch\fall_detection_backend
conda run -n "yueyeu's_project" python manage.py train-fusion
conda run -n "yueyeu's_project" python manage.py train-stress
```

训练产物：

- `models/fusion/activity_context_model.joblib`
- `models/fusion/hr_reference.json`
- `models/fusion/stress_classifier.joblib`
- `output/fusion_training_report.json`
- `output/stress_training_report.json`

## 启动服务

```powershell
cd E:\moon\TeamResearch\fall_detection_backend
conda run -n "yueyeu's_project" python manage.py serve-fusion
```

## API

- `GET /`
- `POST /fusion/ingest`
- `GET /fusion/state/{user_id}`
- `GET /fusion/alerts/{user_id}`

`/fusion/ingest` 示例：

```json
{
  "user_id": "user001",
  "timestamp": "2026-03-11T18:00:00",
  "heart_rate": 92,
  "wrist_acc": [0.2, 9.8, -0.4],
  "wrist_gyro": [0.01, -0.03, 0.02],
  "ppg_quality": 0.95,
  "wrist_bvp": 0.012,
  "wrist_eda": 0.37,
  "wrist_temp": 33.4,
  "context_override": "resting"
}
```

说明：

- `wrist_bvp/wrist_eda/wrist_temp` 建议提供，可提升压力监督模型效果。
- 若不提供，系统会回退到规则评估。
- 冷启动策略默认开启：当历史不足主窗口（120）但达到短窗（48）时，仍执行监督推理。

## 端到端演示

```powershell
cd E:\moon\TeamResearch\fall_detection_backend
conda run -n "yueyeu's_project" python manage.py demo-fusion
conda run -n "yueyeu's_project" python manage.py demo-stress
```

输出目录：`E:\moon\TeamResearch\output`
