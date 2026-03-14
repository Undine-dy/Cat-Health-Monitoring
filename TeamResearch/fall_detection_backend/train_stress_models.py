from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX2,FMA3,AVX512F")

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from project_config import FUSION_MODEL_DIR, OUTPUT_DIR, WESAD_ROOT
from stress_features import build_wesad_window_dataset, stress_feature_names


DATASET_ROOT = WESAD_ROOT
MODEL_DIR = FUSION_MODEL_DIR

WINDOW_SIZE = 120
STEP_SIZE = 20
MIN_LABEL_PURITY = 0.75
SAMPLE_RATE_HZ = 4.0
COLD_START_WINDOW_SIZE = 48
RANDOM_STATE = 42


def _model_candidates() -> list[dict]:
    return [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 600},
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8, "n_estimators": 800},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 700},
        {"max_depth": 6, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0, "n_estimators": 900},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "n_estimators": 900},
    ]


def _build_model(num_class: int, params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        n_estimators=int(params["n_estimators"]),
        reg_lambda=1.2,
        min_child_weight=1.0,
    )


def _evaluate_split(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = model.predict(x)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def _stratified_examples(
    model: XGBClassifier,
    label_encoder: LabelEncoder,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_per_label: int = 5,
) -> list[dict]:
    probs = model.predict_proba(x_test)
    pred = np.argmax(probs, axis=1)
    rng = np.random.default_rng(RANDOM_STATE)
    examples: list[dict] = []
    for class_idx in np.unique(y_test):
        class_indices = np.where(y_test == class_idx)[0]
        if class_indices.size == 0:
            continue
        chosen = rng.choice(class_indices, size=min(sample_per_label, class_indices.size), replace=False)
        for idx in chosen:
            expected = str(label_encoder.inverse_transform([int(y_test[idx])])[0])
            predicted = str(label_encoder.inverse_transform([int(pred[idx])])[0])
            examples.append(
                {
                    "index": int(idx),
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": round(float(np.max(probs[idx])), 6),
                    "match": bool(expected == predicted),
                }
            )
    return examples


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[stage] build_dataset", flush=True)
    x, y, subjects, dataset_meta = build_wesad_window_dataset(
        dataset_root=DATASET_ROOT,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        min_label_purity=MIN_LABEL_PURITY,
        sample_rate_hz=SAMPLE_RATE_HZ,
        verbose=True,
        subject_ids=None,
    )
    print("[stage] dataset_ready", x.shape, flush=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    x_train_val, x_test, y_train_val, y_test, subject_train_val, subject_test = train_test_split(
        x,
        y_encoded,
        subjects,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    x_train, x_val, y_train, y_val, subject_train, subject_val = train_test_split(
        x_train_val,
        y_train_val,
        subject_train_val,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    reports: list[dict] = []
    best_meta: dict | None = None
    best_key = (-1.0, -1.0)
    print("[stage] train_candidates", flush=True)
    for params in _model_candidates():
        print(f"[candidate] {json.dumps(params, ensure_ascii=False)}", flush=True)
        model = _build_model(num_class=len(label_encoder.classes_), params=params)
        model.fit(x_train, y_train, verbose=False)
        val_acc, val_f1 = _evaluate_split(model, x_val, y_val)
        test_acc, test_f1 = _evaluate_split(model, x_test, y_test)
        row = {
            "params": params,
            "validation_accuracy": round(val_acc, 6),
            "validation_macro_f1": round(val_f1, 6),
            "test_accuracy": round(test_acc, 6),
            "test_macro_f1": round(test_f1, 6),
        }
        reports.append(row)
        key = (val_f1, val_acc)
        if key > best_key:
            best_key = key
            best_meta = row

    if best_meta is None:
        raise RuntimeError("No model candidate succeeded")

    final_model = _build_model(num_class=len(label_encoder.classes_), params=dict(best_meta["params"]))
    final_model.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]), verbose=False)

    y_test_pred = final_model.predict(x_test)
    y_test_prob = final_model.predict_proba(x_test)
    labels_order = list(range(len(label_encoder.classes_)))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    report = classification_report(
        y_test,
        y_test_pred,
        labels=labels_order,
        target_names=[str(name) for name in label_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=labels_order).tolist()

    model_bundle = {
        "model": final_model,
        "cold_start_model": None,
        "label_encoder": label_encoder,
        "feature_names": stress_feature_names(),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "cold_start_window_size": COLD_START_WINDOW_SIZE,
        "input_columns": ["acc_x", "acc_y", "acc_z", "acc_mag", "bvp", "eda", "temp"],
        "labels": [str(name) for name in label_encoder.classes_],
    }
    stress_model_path = MODEL_DIR / "stress_classifier.joblib"
    joblib.dump(model_bundle, stress_model_path)

    summary = {
        "dataset": dataset_meta,
        "split_counts": {"train": int(x_train.shape[0]), "validation": int(x_val.shape[0]), "test": int(x_test.shape[0])},
        "split_ratios": {"train": 0.7, "validation": 0.1, "test": 0.2},
        "labels": [str(name) for name in label_encoder.classes_],
        "subject_coverage": {
            "train_unique_subjects": sorted(set(subject_train.tolist())),
            "validation_unique_subjects": sorted(set(subject_val.tolist())),
            "test_unique_subjects": sorted(set(subject_test.tolist())),
        },
        "candidates": reports,
        "best_candidate_by_validation": best_meta,
        "final_test": {
            "accuracy": round(test_accuracy, 6),
            "macro_f1": round(test_macro_f1, 6),
            "avg_confidence": round(float(np.mean(np.max(y_test_prob, axis=1))), 6),
            "classification_report": report,
            "confusion_matrix": cm,
            "label_order": [str(name) for name in label_encoder.classes_],
            "stratified_examples": _stratified_examples(final_model, label_encoder, x_test, y_test),
        },
        "cold_start_policy": {
            "enabled": True,
            "window_size": COLD_START_WINDOW_SIZE,
            "mode": "pad_to_primary_window",
            "note": "When history < primary window and >= cold-start window, engine pads to full window and still uses supervised model.",
        },
        "artifacts": {"stress_model_path": str(stress_model_path)},
    }
    (OUTPUT_DIR / "stress_training_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# WESAD 压力分类训练报告",
        "",
        f"- 数据集路径: `{DATASET_ROOT}`",
        f"- 样本数: {dataset_meta['samples']}, 特征维度: {dataset_meta['feature_dim']}",
        f"- 主模型窗口: {WINDOW_SIZE}, 冷启动窗口: {COLD_START_WINDOW_SIZE}",
        f"- Test Accuracy: {test_accuracy:.4f}",
        f"- Test Macro F1: {test_macro_f1:.4f}",
    ]
    (OUTPUT_DIR / "stress_training_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "test_accuracy": round(test_accuracy, 6),
                "test_macro_f1": round(test_macro_f1, 6),
                "cold_start_window_size": COLD_START_WINDOW_SIZE,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
