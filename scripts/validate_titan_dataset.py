#!/usr/bin/env python3
"""
Optional script to validate the layout of a dataset for 360-VIO processing.
Checks for monotonic timestamps in images/, cam_timestamps.txt, and imu_data.csv.
Checks for the median IMU samples per frame interval.
Checks for the median |gyro| and |accel| norms.
Checks for the IMU rate.
Checks for the camera rate.
Checks for the IMU samples per frame interval.

run with:
    python3 scripts/validate_titan_dataset.py /path/to/sequence

"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_timestamps(path: Path) -> List[float]:
    stamps: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            stamps.append(float(raw))
    return stamps


def load_imu(path: Path) -> List[Tuple[float, float, float, float, float, float, float]]:
    rows: List[Tuple[float, float, float, float, float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        # app/main.cpp unconditionally skips one header row
        next(reader, None)
        for row in reader:
            if len(row) != 7:
                continue
            rows.append(tuple(float(v) for v in row))  # type: ignore[arg-type]
    return rows


def mean_rate(stamps: List[float]) -> float:
    if len(stamps) < 2:
        return 0.0
    dt = stamps[-1] - stamps[0]
    if dt <= 0.0:
        return 0.0
    return len(stamps) / dt


def monotonic(stamps: List[float]) -> bool:
    return all(b > a for a, b in zip(stamps[:-1], stamps[1:]))


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def imu_samples_per_frame(imu_stamps: List[float], cam_stamps: List[float]) -> Tuple[float, int]:
    if len(cam_stamps) < 2 or not imu_stamps:
        return 0.0, 0
    counts: List[int] = []
    zero_bins = 0
    idx = 0
    for i in range(1, len(cam_stamps)):
        start_t = cam_stamps[i - 1]
        end_t = cam_stamps[i]
        count = 0
        while idx < len(imu_stamps) and imu_stamps[idx] < end_t:
            if imu_stamps[idx] >= start_t:
                count += 1
            idx += 1
        counts.append(count)
        if count == 0:
            zero_bins += 1
    return median([float(c) for c in counts]), zero_bins


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset for 360-VIO.")
    parser.add_argument("dataset_dir", type=Path, help="Path to sequence directory.")
    args = parser.parse_args()

    seq = args.dataset_dir
    images_dir = seq / "images"
    cam_file = seq / "cam_timestamps.txt"
    imu_file = seq / "imu_data.csv"

    errors: List[str] = []
    warns: List[str] = []

    if not images_dir.is_dir():
        errors.append(f"Missing images dir: {images_dir}")
    if not cam_file.is_file():
        errors.append(f"Missing camera timestamps file: {cam_file}")
    if not imu_file.is_file():
        errors.append(f"Missing IMU csv file: {imu_file}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return 1

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        errors.append("No images found under images/.")

    # Lexicographic sorting should already be zero-padded for temporal ordering.
    if images and [p.name for p in images] != sorted(p.name for p in images):
        warns.append("Image file names are not lexicographically sortable by time.")

    try:
        cam_stamps = load_timestamps(cam_file)
    except Exception as exc:
        errors.append(f"Failed reading camera timestamps: {exc}")
        cam_stamps = []

    try:
        imu_rows = load_imu(imu_file)
    except Exception as exc:
        errors.append(f"Failed reading IMU csv: {exc}")
        imu_rows = []

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return 1

    imu_stamps = [r[0] for r in imu_rows]
    gyro_norms = [math.sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2) for r in imu_rows]
    accel_norms = [math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in imu_rows]

    print("=== Dataset Summary ===")
    print(f"Sequence: {seq}")
    print(f"Images: {len(images)}")
    print(f"Camera timestamps: {len(cam_stamps)}")
    print(f"IMU rows: {len(imu_rows)}")

    if len(images) != len(cam_stamps):
        warns.append(
            "Image count != camera timestamp count; app truncates to min count."
        )

    if not monotonic(cam_stamps):
        errors.append("Camera timestamps are not strictly increasing.")
    if not monotonic(imu_stamps):
        errors.append("IMU timestamps are not strictly increasing.")

    cam_hz = mean_rate(cam_stamps)
    imu_hz = mean_rate(imu_stamps)
    print(f"Estimated camera rate: {cam_hz:.2f} Hz")
    print(f"Estimated IMU rate: {imu_hz:.2f} Hz")

    median_imu_per_frame, zero_bins = imu_samples_per_frame(imu_stamps, cam_stamps)
    print(f"Median IMU samples per frame interval: {median_imu_per_frame:.1f}")

    if zero_bins > 0:
        warns.append(f"{zero_bins} camera intervals have zero IMU samples.")

    # Heuristics for unit sanity
    gyro_med = median(gyro_norms)
    accel_med = median(accel_norms)
    print(f"Median |gyro|: {gyro_med:.4f}")
    print(f"Median |accel|: {accel_med:.4f}")

    if accel_med < 4.0 or accel_med > 15.0:
        warns.append("Accel magnitude looks unusual for m/s^2 units.")

    # Deg/s data during regular motion often appears too large in rad/s terms.
    if gyro_med > 8.0:
        warns.append("Gyro magnitude is high; verify gyro is in rad/s (not deg/s).")

    if imu_hz < 200.0:
        warns.append("IMU rate appears low for this pipeline; verify capture/export.")

    if errors:
        print("=== FAIL ===")
        for err in errors:
            print(f"[ERROR] {err}")
        return 1

    if warns:
        print("=== WARNINGS ===")
        for warn in warns:
            print(f"[WARN] {warn}")
    else:
        print("=== PASS ===")
        print("Dataset shape looks compatible with app/main.cpp expectations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
