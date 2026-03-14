from pathlib import Path

import numpy as np
import pandas as pd


PAMAP2_COLUMNS = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_acc6_x",
    "hand_acc6_y",
    "hand_acc6_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_1",
    "hand_orient_2",
    "hand_orient_3",
    "hand_orient_4",
    "chest_temp",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_acc6_x",
    "chest_acc6_y",
    "chest_acc6_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_1",
    "chest_orient_2",
    "chest_orient_3",
    "chest_orient_4",
    "ankle_temp",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
    "ankle_acc6_x",
    "ankle_acc6_y",
    "ankle_acc6_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_1",
    "ankle_orient_2",
    "ankle_orient_3",
    "ankle_orient_4",
]

ACTIVITY_TO_CONTEXT = {
    1: "resting",
    2: "resting",
    3: "resting",
    9: "resting",
    10: "resting",
    11: "resting",
    4: "walking",
    7: "walking",
    5: "running",
    20: "running",
    24: "running",
    6: "cycling",
    12: "stairs",
    13: "stairs",
    16: "household",
    17: "household",
    18: "household",
    19: "household",
}

RAW_SIGNAL_COLUMNS = [
    "heart_rate",
    "wrist_acc_x",
    "wrist_acc_y",
    "wrist_acc_z",
    "wrist_gyro_x",
    "wrist_gyro_y",
    "wrist_gyro_z",
]

RAW_SIGNAL_WITH_MAG_COLUMNS = RAW_SIGNAL_COLUMNS + ["wrist_acc_mag", "wrist_gyro_mag"]
STAT_NAMES = ["mean", "std", "min", "max", "median", "q25", "q75", "rms", "energy", "mad"]
CORRELATION_PAIRS = [
    ("wrist_acc_x", "wrist_acc_y"),
    ("wrist_acc_x", "wrist_acc_z"),
    ("wrist_acc_y", "wrist_acc_z"),
    ("wrist_gyro_x", "wrist_gyro_y"),
    ("wrist_gyro_x", "wrist_gyro_z"),
    ("wrist_gyro_y", "wrist_gyro_z"),
]
FREQ_COLUMNS = ["wrist_acc_mag", "wrist_gyro_mag"]


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


def relevant_pamap2_files(dataset_root: Path) -> list[Path]:
    files = list((dataset_root / "Protocol").glob("subject*.dat"))
    files.extend((dataset_root / "Optional").glob("subject*.dat"))
    return sorted(files)


def feature_names() -> list[str]:
    names: list[str] = []
    for column in RAW_SIGNAL_WITH_MAG_COLUMNS:
        for stat in STAT_NAMES:
            names.append(f"{column}__{stat}")
    for left, right in CORRELATION_PAIRS:
        names.append(f"corr__{left}__{right}")
    for column in FREQ_COLUMNS:
        names.append(f"{column}__dominant_bin")
        names.append(f"{column}__dominant_power")
    return names


def load_subject_frame(path: Path) -> pd.DataFrame:
    usecols = [0, 1, 2, 7, 8, 9, 10, 11, 12]
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        na_values="NaN",
        usecols=usecols,
    )
    frame = frame[frame["activity_id"].isin(ACTIVITY_TO_CONTEXT)].copy()
    if frame.empty:
        return frame

    frame = frame.rename(
        columns={
            "hand_acc6_x": "wrist_acc_x",
            "hand_acc6_y": "wrist_acc_y",
            "hand_acc6_z": "wrist_acc_z",
            "hand_gyro_x": "wrist_gyro_x",
            "hand_gyro_y": "wrist_gyro_y",
            "hand_gyro_z": "wrist_gyro_z",
        }
    )
    frame["context"] = frame["activity_id"].map(ACTIVITY_TO_CONTEXT)
    frame["heart_rate"] = frame["heart_rate"].interpolate(limit_direction="both").ffill().bfill()
    frame = frame.dropna(subset=RAW_SIGNAL_COLUMNS).copy()
    frame["wrist_acc_mag"] = np.sqrt(
        frame["wrist_acc_x"] ** 2 + frame["wrist_acc_y"] ** 2 + frame["wrist_acc_z"] ** 2
    )
    frame["wrist_gyro_mag"] = np.sqrt(
        frame["wrist_gyro_x"] ** 2 + frame["wrist_gyro_y"] ** 2 + frame["wrist_gyro_z"] ** 2
    )
    frame["subject_id"] = path.stem
    return frame


def extract_window_features(window: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for idx in range(window.shape[1]):
        signal = window[:, idx]
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
                float(np.sqrt(np.mean(signal ** 2))),
                float(np.mean(signal ** 2)),
                float(np.mean(np.abs(centered))),
            ]
        )

    column_index = {name: idx for idx, name in enumerate(RAW_SIGNAL_WITH_MAG_COLUMNS)}
    for left, right in CORRELATION_PAIRS:
        left_idx = column_index[left]
        right_idx = column_index[right]
        corr = _safe_correlation(window[:, left_idx], window[:, right_idx])
        values.append(corr)

    for column in FREQ_COLUMNS:
        signal = window[:, column_index[column]]
        centered = signal - float(np.mean(signal))
        spectrum = np.abs(np.fft.rfft(centered))
        dominant_bin = int(np.argmax(spectrum[1:]) + 1) if spectrum.shape[0] > 1 else 0
        dominant_power = float(spectrum[dominant_bin]) if dominant_bin < len(spectrum) else 0.0
        values.extend([float(dominant_bin), dominant_power])

    return np.asarray(values, dtype=np.float32)


def build_window_dataset(
    dataset_root: Path,
    window_size: int = 256,
    step_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = step_size or window_size
    features: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []

    for path in relevant_pamap2_files(dataset_root):
        frame = load_subject_frame(path)
        if frame.empty:
            continue
        segment_id = (frame["context"] != frame["context"].shift()).cumsum()
        for _, segment in frame.groupby(segment_id):
            label = str(segment["context"].iloc[0])
            values = segment[RAW_SIGNAL_WITH_MAG_COLUMNS].to_numpy(dtype=np.float32)
            if values.shape[0] < window_size:
                continue
            for start in range(0, values.shape[0] - window_size + 1, step):
                window = values[start : start + window_size]
                features.append(extract_window_features(window))
                labels.append(label)
                subjects.append(str(segment["subject_id"].iloc[0]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(subjects),
    )


def build_hr_reference(dataset_root: Path) -> dict:
    frames = [load_subject_frame(path) for path in relevant_pamap2_files(dataset_root)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError(f"No PAMAP2 frames found in {dataset_root}")
    dataset = pd.concat(frames, ignore_index=True)

    references: dict[str, dict[str, float]] = {}
    for context, group in dataset.groupby("context"):
        values = group["heart_rate"].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        references[str(context)] = {
            "median": round(median, 4),
            "mad": round(max(mad, 1.0), 4),
            "p05": round(float(np.percentile(values, 5)), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "count": int(values.shape[0]),
        }

    references["defaults"] = {
        "sleep_target_hours": 7.5,
        "resting_hr": references.get("resting", {}).get("median", 68.0),
    }
    return references
