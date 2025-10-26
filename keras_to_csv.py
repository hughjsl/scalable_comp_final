
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import tensorflow as tf

INPUT_DIR     = Path("real_captchas")
OUTPUT_CSV    = Path("submission_keras.csv")
KERAS_DIR     = Path("checkpoints/ctc_ocr_keras")
CHARSET_JSON  = Path("checkpoints/charset.json")

IMG_W, IMG_H = 192, 96

from image_processing import PreprocConfig, preprocess_pipeline, load_rgb

def hexish_sort_key(name: str):
    order = {c: i for i, c in enumerate("0123456789abcdef")}
    return [order.get(ch, 100 + ord(ch)) for ch in name.lower()]

def load_meta(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    idx_to_char = j["idx_to_char"]
    blank_idx = int(j["blank_idx"])
    return idx_to_char, blank_idx

def preprocess_for_model(img_path: Path) -> np.ndarray:

    cfg = PreprocConfig()
    rgb = load_rgb(img_path)
    _, _, bw_clean, _, _ = preprocess_pipeline(rgb, cfg)
    if bw_clean.shape != (IMG_H, IMG_W):
        bw_clean = np.array(Image.fromarray(bw_clean).resize((IMG_W, IMG_H), Image.NEAREST), dtype=np.uint8)
    return bw_clean

def to_keras_input(arr_u8: np.ndarray) -> np.ndarray:
    g = arr_u8.astype(np.float32) / 255.0
    return g[None, ..., None]  # [1,H,W,1]

def greedy_decode_batch(logits: np.ndarray, blank_idx: int, idx_to_char: List[str]) -> List[str]:
    # logits: [B,T,C]
    seqs = logits.argmax(axis=-1)  # [B,T]
    outs = []
    for seq in seqs:
        out, prev = [], None
        for p in seq:
            p = int(p)
            if p == blank_idx:
                prev = p; continue
            if prev != p:
                out.append(idx_to_char[p] if p < len(idx_to_char) else "?")
            prev = p
        outs.append("".join(out))
    return outs

def main():
    if not INPUT_DIR.exists():
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
            f.write("hstaffor\n")
        return

    files = sorted([p for p in INPUT_DIR.glob("*.png")], key=lambda p: hexish_sort_key(p.name))
    if not files:
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
            f.write("hstaffor\n")
        return

    idx_to_char, blank_idx = load_meta(CHARSET_JSON)
    model = tf.keras.models.load_model(KERAS_DIR, compile=False)

    print(f"Processing {len(files)} images.\n")
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("hstaffor\n")
        for i, p in enumerate(files, 1):
            try:
                arr = preprocess_for_model(p)
                x   = to_keras_input(arr)
                logits = model(x, training=False).numpy()
                pred = greedy_decode_batch(logits, blank_idx, idx_to_char)[0]
            except Exception:
                pred = ""
            f.write(f"{p.name},{pred}\n")
            if i % 50 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {p.name} → {pred}")

if __name__ == "__main__":
    main()
