import json

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from activity_context_features import (
    build_hr_reference,
    build_window_dataset,
    feature_names,
)
from project_config import FUSION_MODEL_DIR, OUTPUT_DIR, PAMAP2_ROOT


DATASET_ROOT = PAMAP2_ROOT
MODEL_DIR = FUSION_MODEL_DIR

WINDOW_SIZE = 256
WINDOW_DURATION_SECONDS = 2.56


def report_for_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    labels = sorted(np.unique(y_train))
    return {
        "name": name,
        "model": model,
        "val_accuracy": round(float(accuracy_score(y_val, val_pred)), 6),
        "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 6),
        "test_report": classification_report(
            y_test,
            test_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=labels).tolist(),
        "label_order": labels,
    }


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, subjects = build_window_dataset(
        dataset_root=DATASET_ROOT,
        window_size=WINDOW_SIZE,
        step_size=WINDOW_SIZE,
    )
    split_info = {
        "train": 0.7,
        "validation": 0.1,
        "test": 0.2,
    }

    X_trainval, X_test, y_trainval, y_test, subjects_trainval, subjects_test = train_test_split(
        X,
        y,
        subjects,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, subjects_train, subjects_val = train_test_split(
        X_trainval,
        y_trainval,
        subjects_trainval,
        test_size=0.125,
        random_state=42,
        stratify=y_trainval,
    )

    candidates = [
        (
            "svc_rbf_c3_g002",
            make_pipeline(
                StandardScaler(),
                SVC(C=3.0, gamma=0.02, kernel="rbf"),
            ),
        ),
        (
            "svc_rbf_c5_scale",
            make_pipeline(
                StandardScaler(),
                SVC(C=5.0, gamma="scale", kernel="rbf"),
            ),
        ),
        (
            "knn_distance",
            make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5, weights="distance"),
            ),
        ),
    ]

    reports = [
        report_for_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test)
        for name, model in candidates
    ]
    reports.sort(key=lambda item: (item["val_accuracy"], item["test_accuracy"]), reverse=True)
    best = reports[0]

    final_model = candidates[[name for name, _ in candidates].index(best["name"])][1]
    final_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    model_bundle = {
        "model": final_model,
        "model_type": "windowed_classifier",
        "window_size": WINDOW_SIZE,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "raw_feature_names": [
            "heart_rate",
            "wrist_acc_x",
            "wrist_acc_y",
            "wrist_acc_z",
            "wrist_gyro_x",
            "wrist_gyro_y",
            "wrist_gyro_z",
        ],
        "window_feature_names": feature_names(),
        "context_labels": sorted(np.unique(y)),
        "best_model_name": best["name"],
        "split_info": split_info,
    }
    joblib.dump(model_bundle, MODEL_DIR / "activity_context_model.joblib")

    hr_reference = build_hr_reference(DATASET_ROOT)
    (MODEL_DIR / "hr_reference.json").write_text(
        json.dumps(hr_reference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "window_size": WINDOW_SIZE,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "split_counts": {
            "train": int(X_train.shape[0]),
            "validation": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "split_ratios": split_info,
        "label_distribution": {label: int(count) for label, count in zip(*np.unique(y, return_counts=True))},
        "models": [
            {
                "name": report["name"],
                "val_accuracy": report["val_accuracy"],
                "test_accuracy": report["test_accuracy"],
            }
            for report in reports
        ],
        "best_model": {
            "name": best["name"],
            "val_accuracy": best["val_accuracy"],
            "test_accuracy": best["test_accuracy"],
            "test_report": best["test_report"],
            "test_confusion_matrix": best["test_confusion_matrix"],
            "label_order": best["label_order"],
        },
        "subject_coverage": {
            "train_unique_subjects": sorted(set(subjects_train.tolist())),
            "validation_unique_subjects": sorted(set(subjects_val.tolist())),
            "test_unique_subjects": sorted(set(subjects_test.tolist())),
        },
    }
    (OUTPUT_DIR / "fusion_training_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "best_model": best["name"],
                "validation_accuracy": best["val_accuracy"],
                "test_accuracy": best["test_accuracy"],
                "samples": int(X.shape[0]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
