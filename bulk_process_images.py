
from pathlib import Path
from typing import List, Tuple
import csv
import json
import shutil
import sys

import numpy as np
import cv2


from image_processing import PreprocConfig, preprocess_pipeline

IN_DIR  = Path("output_captchas")
OUT_DIR = Path("processed_captchas")

REPORT_CSV_NAME = "preprocess_report.csv"

OVERWRITE = True

N_WORKERS = 1

GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg")

CFG = PreprocConfig(
    ADAPT_BLOCK=31,
    ADAPT_C=7,
    ERODE_AFTER_BIN=True,
    ERODE_KERNEL=(2, 2),
    DOT_MAX_W=8,
    DOT_MAX_H=8,
    DOT_MAX_AREA=49,
    DOT_MARGIN=2,
    DOT_ALLOW_RING_FG=2,
    MIN_W=5, MIN_H=10,
    UNDERLINE_WHR=4.0,
)

def list_images(folder: Path, patterns: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(folder.glob(pat))
    return sorted(files)

def boxes_to_str(boxes: List[Tuple[int,int,int,int]]) -> str:
    return "|".join(f"{x} {y} {w} {h}" for (x,y,w,h) in boxes)

def process_one(path: Path) -> Tuple[str, int, int, str]:

    rgb, dark, bw_clean, boxes, removed = preprocess_pipeline(path, CFG)

    out_path = OUT_DIR / path.name
    if OVERWRITE or not out_path.exists():
        ok = cv2.imwrite(str(out_path), bw_clean)
        if not ok:
            raise RuntimeError(f"Failed to save: {out_path}")

    return path.name, int(removed), int(len(boxes)), boxes_to_str(boxes)

def main():
    if not IN_DIR.exists():
        raise SystemExit(f"Input folder not found: {IN_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list_images(IN_DIR, GLOB_PATTERNS)
    if not files:
        raise SystemExit(f"No images found in {IN_DIR.resolve()} (patterns: {GLOB_PATTERNS}).")

    labels_src = IN_DIR / "labels.csv"
    labels_dst = OUT_DIR / "labels.csv"
    if labels_src.exists():
        if OVERWRITE or not labels_dst.exists():
            shutil.copy2(labels_src, labels_dst)
            print(f"Copied labels.csv to {labels_dst}")
    else:
        print("No labels.csv found in input")

    print(f"Processing {len(files)} images from {IN_DIR} -> {OUT_DIR} ...")

    report_rows: List[Tuple[str,int,int,str]] = []
    for i, p in enumerate(files, 1):
        try:
            row = process_one(p)
            report_rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed on {p.name}: {e}", file=sys.stderr)
        if i % 50 == 0 or i == len(files):
            print(f"  done {i}/{len(files)}")

    report_csv = OUT_DIR / REPORT_CSV_NAME
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "removed_dots", "n_boxes", "boxes"])
        for r in report_rows:
            w.writerow(r)

    print("Done.")

if __name__ == "__main__":
    main()
