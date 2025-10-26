
import json
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

INPUT_DIR    = Path("real_captchas")
OUTPUT_CSV   = Path("submission_tflite.csv")
TFLITE_MODEL = Path("checkpoints/ctc_ocr_fp16.tflite")
CHARSET_JSON = Path("checkpoints/charset.json")


IMG_W, IMG_H = 192, 96


from image_processing import PreprocConfig, preprocess_pipeline, load_rgb

def hexish_sort_key(name: str):
    order = {c:i for i, c in enumerate("0123456789abcdef")}
    return [order.get(ch, 100 + ord(ch)) for ch in name.lower()]

with open(CHARSET_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)
idx_to_char = meta["idx_to_char"]
blank_idx = int(meta["blank_idx"])

def preprocess_for_model(img_path: Path) -> np.ndarray:
    cfg = PreprocConfig()
    rgb = load_rgb(img_path)
    _, _, bw_clean, _, _ = preprocess_pipeline(rgb, cfg)     # HxW uint8
    if bw_clean.shape != (IMG_H, IMG_W):
        bw_clean = np.array(Image.fromarray(bw_clean).resize((IMG_W, IMG_H), Image.NEAREST), dtype=np.uint8)
    g = bw_clean.astype(np.float32) / 255.0
    return g[None, ..., None]  # [1,H,W,1]

def greedy_decode(logits: np.ndarray) -> str:
    seq = logits[0].argmax(-1)
    out, prev = [], None
    for p in seq:
        p = int(p)
        if p == blank_idx:
            prev = p; continue
        if prev != p:
            out.append(idx_to_char[p] if p < len(idx_to_char) else "?")
        prev = p
    return "".join(out)

def main():
    assert TFLITE_MODEL.is_file(), f"Not a file: {TFLITE_MODEL}"
    files = sorted(INPUT_DIR.glob("*.png"), key=lambda p: hexish_sort_key(p.name))
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("hstaffor\n")
        interp = tflite.Interpreter(model_path=str(TFLITE_MODEL))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]

        for i, p in enumerate(files, 1):
            try:
                x = preprocess_for_model(p).astype(np.float32)
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                logits = interp.get_tensor(out["index"])  # [1,T,C]
                pred = greedy_decode(logits)
            except Exception:
                pred = ""
            f.write(f"{p.name},{pred}\n")
            if i % 50 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {p.name} → {pred}")

if __name__ == "__main__":
    main()
