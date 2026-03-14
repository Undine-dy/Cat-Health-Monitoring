from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAMAP2_ROOT = DATASET_ROOT / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset"
WESAD_ROOT = DATASET_ROOT / "WESAD_Kaggle" / "WESAD"
OUTPUT_ROOT = PROJECT_ROOT / "New_dataset"

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
    4: "walking",
    5: "running",
    6: "cycling",
    7: "walking",
    9: "resting",
    10: "resting",
    11: "resting",
    12: "stairs",
    13: "stairs",
    16: "household",
    17: "household",
    18: "household",
    19: "household",
    20: "running",
    24: "running",
}

WESAD_LABEL_MAP = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == target_len:
        return arr.astype(np.float32, copy=False)
    idx = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.int64)
    return arr[idx].astype(np.float32, copy=False)


def _load_pamap2_subject(path: Path, downsample_step: int) -> pd.DataFrame:
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
    frame["heart_rate"] = frame["heart_rate"].interpolate(limit_direction="both").ffill().bfill()
    frame = frame.dropna(
        subset=["heart_rate", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z"]
    ).copy()
    if downsample_step > 1:
        frame = frame.iloc[::downsample_step].copy()
    frame["activity_context"] = frame["activity_id"].map(ACTIVITY_TO_CONTEXT)
    frame["source_subject"] = path.stem
    frame = frame.reset_index(drop=True)
    return frame[
        [
            "source_subject",
            "timestamp",
            "activity_id",
            "activity_context",
            "heart_rate",
            "hand_acc6_x",
            "hand_acc6_y",
            "hand_acc6_z",
            "hand_gyro_x",
            "hand_gyro_y",
            "hand_gyro_z",
        ]
    ]


def _load_wesad_subject(path: Path, downsample_step: int) -> pd.DataFrame:
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    wrist = payload["signal"]["wrist"]
    labels_raw = np.asarray(payload["label"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1, 1).astype(np.float32, copy=False)
    target_len = int(eda.shape[0])
    if target_len == 0:
        return pd.DataFrame()
    acc = _resample_to_length(np.asarray(wrist["ACC"]), target_len)
    bvp = _resample_to_length(np.asarray(wrist["BVP"]), target_len).reshape(-1)
    temp = _resample_to_length(np.asarray(wrist["TEMP"]), target_len).reshape(-1)
    labels = _resample_to_length(labels_raw, target_len).reshape(-1).astype(np.int32, copy=False)
    acc_mag = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
    frame = pd.DataFrame(
        {
            "source_subject": path.parent.name,
            "stress_index": np.arange(target_len, dtype=np.int64),
            "stress_label_id": labels,
            "stress_acc_x": acc[:, 0],
            "stress_acc_y": acc[:, 1],
            "stress_acc_z": acc[:, 2],
            "stress_acc_mag": acc_mag,
            "wrist_bvp": bvp,
            "wrist_eda": eda.reshape(-1),
            "wrist_temp": temp,
        }
    )
    frame = frame[frame["stress_label_id"].isin(WESAD_LABEL_MAP)].copy()
    frame["stress_label"] = frame["stress_label_id"].map(WESAD_LABEL_MAP)
    if downsample_step > 1:
        frame = frame.iloc[::downsample_step].copy()
    frame = frame.reset_index(drop=True)
    return frame[
        [
            "source_subject",
            "stress_index",
            "stress_label_id",
            "stress_label",
            "stress_acc_x",
            "stress_acc_y",
            "stress_acc_z",
            "stress_acc_mag",
            "wrist_bvp",
            "wrist_eda",
            "wrist_temp",
        ]
    ]


def _cyclic_slice(frame: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
    idx = (np.arange(length, dtype=np.int64) + int(start)) % len(frame)
    out = frame.iloc[idx].copy().reset_index()
    out = out.rename(columns={"index": "source_row_index"})
    return out


def build_dataset(
    num_people: int,
    rows_per_person: int,
    pamap_step: int,
    wesad_step: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict], dict]:
    rng = np.random.default_rng(seed)

    pamap_files = sorted((PAMAP2_ROOT / "Protocol").glob("subject*.dat"))
    if not pamap_files:
        raise FileNotFoundError(f"No PAMAP2 subject files under {PAMAP2_ROOT}")
    pamap_map: dict[str, pd.DataFrame] = {}
    for path in pamap_files:
        frame = _load_pamap2_subject(path, downsample_step=pamap_step)
        if not frame.empty:
            pamap_map[path.stem] = frame
    if not pamap_map:
        raise RuntimeError("No usable PAMAP2 rows found")

    wesad_files = sorted(WESAD_ROOT.glob("S*/S*.pkl"))
    if not wesad_files:
        raise FileNotFoundError(f"No WESAD subject pkl files under {WESAD_ROOT}")
    wesad_map: dict[str, pd.DataFrame] = {}
    for path in wesad_files:
        frame = _load_wesad_subject(path, downsample_step=wesad_step)
        if not frame.empty:
            wesad_map[path.parent.name] = frame
    if not wesad_map:
        raise RuntimeError("No usable WESAD rows found")

    pamap_subjects = sorted(pamap_map.keys())
    wesad_subjects = sorted(wesad_map.keys())
    timeline_rows: list[dict] = []
    profile_payloads: list[dict] = []

    for i in range(num_people):
        person_id = f"ID_{i + 1:03d}"
        pamap_subject = pamap_subjects[i % len(pamap_subjects)]
        wesad_subject = wesad_subjects[(i * 3) % len(wesad_subjects)]
        pamap_frame = pamap_map[pamap_subject]
        wesad_frame = wesad_map[wesad_subject]

        pamap_start = int(rng.integers(0, len(pamap_frame)))
        wesad_start = int(rng.integers(0, len(wesad_frame)))
        pamap_slice = _cyclic_slice(pamap_frame, start=pamap_start, length=rows_per_person)
        wesad_slice = _cyclic_slice(wesad_frame, start=wesad_start, length=rows_per_person)

        person_records: list[dict] = []
        for step in range(rows_per_person):
            p = pamap_slice.iloc[step]
            w = wesad_slice.iloc[step]
            record = {
                "person_id": person_id,
                "step": int(step),
                "sequence_second": float(p["timestamp"]),
                "activity_id": int(p["activity_id"]),
                "activity_context": str(p["activity_context"]),
                "heart_rate": float(p["heart_rate"]),
                "wrist_acc_x": float(p["hand_acc6_x"]),
                "wrist_acc_y": float(p["hand_acc6_y"]),
                "wrist_acc_z": float(p["hand_acc6_z"]),
                "wrist_gyro_x": float(p["hand_gyro_x"]),
                "wrist_gyro_y": float(p["hand_gyro_y"]),
                "wrist_gyro_z": float(p["hand_gyro_z"]),
                "stress_label_id": int(w["stress_label_id"]),
                "stress_label": str(w["stress_label"]),
                "stress_acc_x": float(w["stress_acc_x"]),
                "stress_acc_y": float(w["stress_acc_y"]),
                "stress_acc_z": float(w["stress_acc_z"]),
                "stress_acc_mag": float(w["stress_acc_mag"]),
                "wrist_bvp": float(w["wrist_bvp"]),
                "wrist_eda": float(w["wrist_eda"]),
                "wrist_temp": float(w["wrist_temp"]),
                "source_pamap2_subject": str(p["source_subject"]),
                "source_pamap2_row_index": int(p["source_row_index"]),
                "source_wesad_subject": str(w["source_subject"]),
                "source_wesad_row_index": int(w["source_row_index"]),
            }
            timeline_rows.append(record)
            person_records.append(record)

        person_frame = pd.DataFrame(person_records)
        profile_payloads.append(
            {
                "person_id": person_id,
                "source_pair": {
                    "pamap2_subject": pamap_subject,
                    "wesad_subject": wesad_subject,
                },
                "overview": {
                    "records": int(len(person_records)),
                    "heart_rate_min": float(person_frame["heart_rate"].min()),
                    "heart_rate_max": float(person_frame["heart_rate"].max()),
                    "heart_rate_mean": float(person_frame["heart_rate"].mean()),
                    "activity_distribution": {
                        str(k): int(v) for k, v in person_frame["activity_context"].value_counts().to_dict().items()
                    },
                    "stress_distribution": {
                        str(k): int(v) for k, v in person_frame["stress_label"].value_counts().to_dict().items()
                    },
                },
                "timeline": person_records,
            }
        )

    timeline = pd.DataFrame(timeline_rows)
    manifest = {
        "project_root": str(PROJECT_ROOT),
        "pamap2_root": str(PAMAP2_ROOT),
        "wesad_root": str(WESAD_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "num_people": int(num_people),
        "rows_per_person": int(rows_per_person),
        "total_rows": int(len(timeline_rows)),
        "pamap_subjects_used": pamap_subjects,
        "wesad_subjects_used": wesad_subjects,
        "sampling": {
            "pamap_downsample_step": int(pamap_step),
            "wesad_downsample_step": int(wesad_step),
            "seed": int(seed),
        },
        "rule": "All numeric values are taken directly from PAMAP2/WESAD rows; only sequence pairing is recombined.",
    }
    return timeline, profile_payloads, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-people", type=int, default=12)
    parser.add_argument("--rows-per-person", type=int, default=360)
    parser.add_argument("--pamap-step", type=int, default=40)
    parser.add_argument("--wesad-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timeline, profiles, manifest = build_dataset(
        num_people=args.num_people,
        rows_per_person=args.rows_per_person,
        pamap_step=args.pamap_step,
        wesad_step=args.wesad_step,
        seed=args.seed,
    )

    timeline_path = OUTPUT_ROOT / "virtual_people_timeline.csv"
    profiles_path = OUTPUT_ROOT / "virtual_people_report_input.json"
    manifest_path = OUTPUT_ROOT / "dataset_manifest.json"

    timeline.to_csv(timeline_path, index=False, encoding="utf-8")
    profiles_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "timeline_path": str(timeline_path),
                "profiles_path": str(profiles_path),
                "manifest_path": str(manifest_path),
                "rows": int(timeline.shape[0]),
                "people": int(args.num_people),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
