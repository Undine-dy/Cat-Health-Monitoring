from __future__ import annotations

import gc
import os
import pickle
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX2,FMA3,AVX512F")

import numpy as np

if hasattr(np, "VisibleDeprecationWarning"):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


TARGET_LABELS = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

RUNTIME_FEATURE_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "acc_mag",
    "bvp",
    "eda",
    "temp",
]

CHANNEL_STAT_NAMES = ["mean", "std", "min", "max", "median", "q25", "q75", "slope", "energy", "mad"]

CORRELATION_PAIRS = [
    ("acc_x", "acc_y"),
    ("acc_x", "acc_z"),
    ("acc_y", "acc_z"),
]


def _percentile_linear(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    if sorted_x.size == 1:
        return float(sorted_x[0])
    rank = float(q) * (sorted_x.size - 1)
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    if low == high:
        return float(sorted_x[low])
    weight = rank - low
    return float(sorted_x[low] * (1.0 - weight) + sorted_x[high] * weight)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.sum(x_centered * y_centered) / denom)
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return corr


def _safe_acc_magnitude(acc_xyz: np.ndarray) -> np.ndarray:
    x = acc_xyz[:, 0]
    y = acc_xyz[:, 1]
    z = acc_xyz[:, 2]
    return np.sqrt(x * x + y * y + z * z).reshape(-1, 1).astype(np.float32, copy=False)


def _safe_slope(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float((x[-1] - x[0]) / max(1, x.size - 1))


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == target_len:
        return arr.astype(np.float32, copy=False)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    indices = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.int64)
    return arr[indices].astype(np.float32, copy=False)


def stress_feature_names() -> list[str]:
    names: list[str] = []
    for column in RUNTIME_FEATURE_COLUMNS:
        for stat_name in CHANNEL_STAT_NAMES:
            names.append(f"{column}__{stat_name}")
    for left, right in CORRELATION_PAIRS:
        names.append(f"corr__{left}__{right}")
    names.extend(
        [
            "bvp__range",
            "bvp__abs_diff_mean",
            "eda__range",
            "temp__range",
        ]
    )
    return names


def _fill_nan_channel(channel: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    finite_mask = np.isfinite(channel)
    if np.all(finite_mask):
        return channel
    if not np.any(finite_mask):
        return np.full_like(channel, float(fallback), dtype=np.float32)
    median = float(np.median(channel[finite_mask]))
    return np.where(finite_mask, channel, median).astype(np.float32, copy=False)


def extract_runtime_stress_features(window: np.ndarray, sample_rate_hz: float = 4.0) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] != len(RUNTIME_FEATURE_COLUMNS):
        raise ValueError(
            f"window shape must be [N, {len(RUNTIME_FEATURE_COLUMNS)}], got {window.shape}"
        )
    values: list[float] = []
    column_index = {name: idx for idx, name in enumerate(RUNTIME_FEATURE_COLUMNS)}
    cleaned = np.asarray(window, dtype=np.float32)

    for idx, column_name in enumerate(RUNTIME_FEATURE_COLUMNS):
        signal = _fill_nan_channel(cleaned[:, idx], fallback=0.0)
        mean = float(np.mean(signal))
        centered = signal - mean
        values.extend(
            [
                mean,
                float(np.std(signal)),
                float(np.min(signal)),
                float(np.max(signal)),
                _percentile_linear(signal, 0.5),
                _percentile_linear(signal, 0.25),
                _percentile_linear(signal, 0.75),
                _safe_slope(signal),
                float(np.mean(signal ** 2)),
                float(np.mean(np.abs(centered))),
            ]
        )

    for left, right in CORRELATION_PAIRS:
        values.append(
            _safe_correlation(
                _fill_nan_channel(cleaned[:, column_index[left]]),
                _fill_nan_channel(cleaned[:, column_index[right]]),
            )
        )

    bvp = _fill_nan_channel(cleaned[:, column_index["bvp"]])
    eda = _fill_nan_channel(cleaned[:, column_index["eda"]])
    temp = _fill_nan_channel(cleaned[:, column_index["temp"]])
    bvp_diff = np.diff(bvp) if bvp.size > 1 else np.asarray([0.0], dtype=np.float32)
    values.extend(
        [
            float(np.max(bvp) - np.min(bvp)),
            float(np.mean(np.abs(bvp_diff))),
            float(np.max(eda) - np.min(eda)),
            float(np.max(temp) - np.min(temp)),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _majority_label(labels: np.ndarray) -> tuple[int | None, float]:
    if labels.size == 0:
        return None, 0.0
    int_labels = labels.astype(np.int32, copy=False)
    max_label = int(np.max(int_labels)) if int_labels.size > 0 else 0
    counts = np.bincount(int_labels, minlength=max(8, max_label + 1))
    best_label = int(np.argmax(counts))
    purity = float(counts[best_label]) / float(labels.size)
    return best_label, purity


def _list_subject_pickles(dataset_root: Path) -> list[Path]:
    candidates = sorted(dataset_root.glob("S*/S*.pkl"))
    return [path for path in candidates if path.is_file()]


def build_wesad_window_dataset(
    dataset_root: Path,
    window_size: int = 120,
    step_size: int = 20,
    min_label_purity: float = 0.75,
    sample_rate_hz: float = 4.0,
    verbose: bool = False,
    subject_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    subject_stats: dict[str, dict[str, int]] = {}

    selected = set(subject_ids) if subject_ids else None
    for pkl_path in _list_subject_pickles(dataset_root):
        if selected is not None and pkl_path.parent.name not in selected:
            continue
        if verbose:
            print(f"[wesad] processing {pkl_path.parent.name}", flush=True)
        with pkl_path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")

        wrist = payload["signal"]["wrist"]
        label_raw = np.asarray(payload["label"]).reshape(-1)
        eda = np.asarray(wrist["EDA"]).reshape(-1, 1).astype(np.float32, copy=False)
        target_len = int(eda.shape[0])
        if target_len < window_size:
            continue

        acc = _resample_to_length(np.asarray(wrist["ACC"]), target_len)
        bvp = _resample_to_length(np.asarray(wrist["BVP"]), target_len)
        temp = _resample_to_length(np.asarray(wrist["TEMP"]), target_len)
        labels_resampled = _resample_to_length(label_raw, target_len).reshape(-1).astype(np.int32, copy=False)

        acc_mag = _safe_acc_magnitude(acc)
        stream = np.hstack([acc, acc_mag, bvp, eda, temp]).astype(np.float32, copy=False)

        subject_id = pkl_path.parent.name
        subject_stats.setdefault(subject_id, {name: 0 for name in TARGET_LABELS.values()})
        max_start = target_len - window_size
        for start in range(0, max_start + 1, step_size):
            end = start + window_size
            window_labels = labels_resampled[start:end]
            label, purity = _majority_label(window_labels)
            if label not in TARGET_LABELS:
                continue
            if purity < min_label_purity:
                continue

            window = stream[start:end, :]
            feature_vec = extract_runtime_stress_features(window=window, sample_rate_hz=sample_rate_hz)
            features.append(feature_vec)
            labels.append(TARGET_LABELS[label])
            subjects.append(subject_id)
            subject_stats[subject_id][TARGET_LABELS[label]] += 1

        del payload, wrist, label_raw, eda, acc, bvp, temp, labels_resampled, acc_mag, stream
        gc.collect()

    if not features:
        raise RuntimeError(f"No usable windows found under {dataset_root}")

    x_array = np.asarray(features, dtype=np.float32)
    y_array = np.asarray(labels)
    s_array = np.asarray(subjects)

    class_names, class_counts = np.unique(y_array, return_counts=True)
    metadata = {
        "dataset_root": str(dataset_root),
        "selected_subjects": sorted(selected) if selected else None,
        "window_size": int(window_size),
        "step_size": int(step_size),
        "sample_rate_hz": float(sample_rate_hz),
        "min_label_purity": float(min_label_purity),
        "samples": int(x_array.shape[0]),
        "feature_dim": int(x_array.shape[1]),
        "class_distribution": {str(name): int(count) for name, count in zip(class_names, class_counts)},
        "subject_stats": subject_stats,
    }
    return x_array, y_array, s_array, metadata
