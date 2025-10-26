import os, csv, json, time, math, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = Path("processed_captchas")
LABELS_CSV = DATA_DIR / "labels.csv"
OUT_DIR = Path("checkpoints"); OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 192, 96

CHARSET = "123456789adeghjknoswxBCFMPQRTUVYZ=#{}[]%+\\|"
BLANK_TOKEN = "<blank>"
SEED = 2025

EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_TRAIN_SAMPLES = 64000
MAX_VAL_SAMPLES   = 6400

AUG_BRIGHTNESS = 0.0
AUG_CONTRAST   = 0.0
AUG_GAUSS_STD  = 0.0

KERAS_PATH      = OUT_DIR / "ctc_ocr_keras"
SAVEDMODEL_DIR  = OUT_DIR / "savedmodel_ctc"
TFLITE_FP32     = OUT_DIR / "ctc_ocr_fp32.tflite"
TFLITE_FP16     = OUT_DIR / "ctc_ocr_fp16.tflite"
CHARSET_JSON    = OUT_DIR / "charset.json"

def set_seed(s: int):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def safe_load_labeled_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f); _ = next(r, None)  # skip header
        for name, label in r:
            rows.append((name, label))
    return rows

def build_vocab(charset: str):
    idx_to_char = list(charset) + [BLANK_TOKEN]
    char_to_idx = {c: i for i, c in enumerate(idx_to_char)}
    blank_idx = len(idx_to_char) - 1
    return char_to_idx, idx_to_char, blank_idx

def encode_label(s: str, char_to_idx: dict) -> np.ndarray:
    return np.array([char_to_idx[c] for c in s], dtype=np.int32)

def cer(hyp: str, ref: str) -> float:
    m, n = len(ref), len(hyp)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            cur = dp[j]
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[n] / max(1, m)

def greedy_decode_logits(logits: np.ndarray, blank_idx: int, idx_to_char: List[str]) -> List[str]:
    preds = logits.argmax(axis=-1)  # [B,T]
    outs = []
    for seq in preds:
        out = []; prev = None
        for p in seq:
            if p == blank_idx:
                prev = p; continue
            if prev != p:
                out.append(idx_to_char[int(p)])
            prev = p
        outs.append("".join(out))
    return outs

def load_and_preprocess(img_path: Path, train: bool) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    if img.size != (IMG_W, IMG_H):
        img = img.resize((IMG_W, IMG_H), Image.NEAREST)
    g = np.asarray(img, dtype=np.float32) / 255.0  # HxW
    if train:
        if AUG_BRIGHTNESS > 0:
            delta = (np.random.rand() * 2 - 1) * AUG_BRIGHTNESS
            g = np.clip(g + delta, 0, 1)
        if AUG_CONTRAST > 0:
            c = 1.0 + ((np.random.rand() * 2 - 1) * AUG_CONTRAST)
            mean = g.mean(); g = np.clip((g - mean) * c + mean, 0, 1)
        if AUG_GAUSS_STD > 0:
            g = np.clip(g + np.random.normal(0.0, AUG_GAUSS_STD, g.shape).astype(np.float32), 0, 1)
    return g[..., None]  # HxW x1

def make_splits(rows, split=0.9):
    rng = random.Random(SEED); rng.shuffle(rows)
    n_train = int(len(rows) * split)
    return rows[:n_train], rows[n_train:]

def tf_dataset(rows, char_to_idx, train: bool, batch_size: int, max_samples: int, blank_idx: int):
    rows = rows[:max_samples] if max_samples else rows

    def gen():
        for name, label in rows:
            img = load_and_preprocess(DATA_DIR / name, train=train)
            y = encode_label(label, char_to_idx)     # variable length (1..6)
            yield {
                "image": img.astype(np.float32),
                "label": y,
                "label_len": np.int32(len(y)),
                "name": name.encode("utf-8"),
                "raw": label.encode("utf-8"),
            }

    output_signature = {
        "image": tf.TensorSpec(shape=(IMG_H, IMG_W, 1), dtype=tf.float32),
        "label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "label_len": tf.TensorSpec(shape=(), dtype=tf.int32),
        "name": tf.TensorSpec(shape=(), dtype=tf.string),
        "raw": tf.TensorSpec(shape=(), dtype=tf.string),
    }
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if train:
        ds = ds.shuffle(4096, seed=SEED)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "image": (IMG_H, IMG_W, 1),
            "label": [None],
            "label_len": [],
            "name": [],
            "raw": [],
        },
        padding_values={
            "image": tf.constant(0.0, tf.float32),
            "label": tf.constant(blank_idx, tf.int32),  # pad = blank
            "label_len": tf.constant(0, tf.int32),
            "name": tf.constant(b"", tf.string),
            "raw": tf.constant(b"", tf.string),
        },
        drop_remainder=train,
    )

    def pack(b):
        return (b["image"], b["label"], b["label_len"], b["name"], b["raw"])
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def build_model(n_classes: int):
    # [B, H, W, 1]
    inp = keras.Input(shape=(IMG_H, IMG_W, 1), name="image")

    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2)(x)     # 96x192 -> 48x96

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2)(x)     # -> 24x48

    x = layers.Conv2D(96, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x) # height/2, keep width -> 12x48

    x = layers.Conv2D(96, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = tf.reduce_mean(x, axis=1)  # [B, T=48, C=96]

    x = layers.Bidirectional(layers.GRU(96, return_sequences=True), merge_mode="concat")(x)  # [B,48,192]
    x = layers.Dropout(0.1)(x)
    logits = layers.Dense(n_classes)(x)  # [B,48,C]

    return keras.Model(inp, logits, name="crnn_ctc_small")

# ---------------- Loss ----------------
class CTCLossLayer(keras.layers.Layer):
    def __init__(self, blank_index: int, **kwargs):
        super().__init__(**kwargs); self.blank_index = blank_index
    def call(self, y_true, y_pred, label_length):
        y_prob = tf.nn.softmax(y_pred, axis=-1)
        batch_size = tf.shape(y_prob)[0]
        T = tf.shape(y_prob)[1]
        input_length = tf.fill([batch_size, 1], T)
        loss = keras.backend.ctc_batch_cost(y_true, y_prob, input_length,
                                            tf.expand_dims(label_length, 1))
        return tf.reduce_mean(loss)

class AdamW(keras.optimizers.Adam):
    def __init__(self, weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs); self.weight_decay = weight_decay
    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.weight_decay and "bias" not in var.name and "batch_normalization" not in var.name:
            var.assign_sub(self.weight_decay * var)
        return super()._resource_apply_dense(grad, var, apply_state)

def tfa_adamw(lr, wd): return AdamW(learning_rate=lr, weight_decay=wd)

class ProgressETA:
    def __init__(self, steps_per_epoch: int, total_epochs: int):
        self.spe = steps_per_epoch
        self.total_epochs = total_epochs
        self.epoch_start = None
        self.batch_start = None
        self.b = 0
    def on_epoch_begin(self, epoch):
        self.epoch_start = time.time()
        self.b = 0
        print(f"\n=== Epoch {epoch+1}/{self.total_epochs} ===")
    def on_train_batch_begin(self, batch_idx: int):
        self.batch_start = time.time()
    def on_train_batch_end(self, batch_idx: int, *, loss: float, bs: int):
        dt = (time.time() - self.batch_start) if self.batch_start else float("nan")
        self.b += 1
        remain = max(self.spe - self.b, 0)
        eta_s = remain * (0 if not np.isfinite(dt) else dt)
        ips = (bs / max(dt, 1e-6)) if np.isfinite(dt) else float("nan")
        print(f"[{self.b:4d}/{self.spe}] {dt:5.2f}s | ETA {eta_s/60:5.1f}m | ~{ips:6.1f} img/s | loss {loss:.4f}",
              end="\r")
    def on_epoch_end(self, epoch):
        ep_min = (time.time() - self.epoch_start) / 60.0 if self.epoch_start else 0.0
        print(f"\nEpoch {epoch+1} finished in {ep_min:.2f} min")

def eval_one_epoch(model, ds, blank_idx, idx_to_char):
    cer_sum = 0.0; exact = 0; n = 0
    for imgs, labels, label_lens, names, raws in ds:
        logits = model(imgs, training=False).numpy()
        hyps = greedy_decode_logits(logits, blank_idx, idx_to_char)
        for h, r in zip(hyps, raws.numpy()):
            ref = r.decode("utf-8")
            cer_sum += cer(h, ref); exact += int(h == ref); n += 1
    return (cer_sum / max(1, n)), (exact / max(1, n))

def export_all(model, idx_to_char, blank_idx):
    model.save(KERAS_PATH, include_optimizer=False)
    tf.saved_model.save(model, str(SAVEDMODEL_DIR))
    meta = {"idx_to_char": idx_to_char, "blank_idx": int(blank_idx), "img_h": IMG_H, "img_w": IMG_W}
    with open(CHARSET_JSON, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    print(f"Saved → {KERAS_PATH} and {SAVEDMODEL_DIR}; meta → {CHARSET_JSON}")

    conv = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))
    with open(TFLITE_FP32, "wb") as f: f.write(conv.convert())
    conv = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    with open(TFLITE_FP16, "wb") as f: f.write(conv.convert())
    print(f"TFLite saved: {TFLITE_FP32} (fp32), {TFLITE_FP16} (fp16-weights)")

def main():
    print(f"TensorFlow: {tf.__version__}")
    set_seed(SEED)

    char_to_idx, idx_to_char, blank_idx = build_vocab(CHARSET)
    with open(CHARSET_JSON, "w", encoding="utf-8") as f:
        json.dump({"idx_to_char": idx_to_char, "blank_idx": int(blank_idx)}, f, indent=2)

    rows = safe_load_labeled_rows(LABELS_CSV)
    train_rows, val_rows = make_splits(rows, split=0.9)

    train_ds = tf_dataset(train_rows, char_to_idx, train=True,
                          batch_size=BATCH_SIZE, max_samples=MAX_TRAIN_SAMPLES, blank_idx=blank_idx)
    val_ds   = tf_dataset(val_rows,   char_to_idx, train=False,
                          batch_size=BATCH_SIZE, max_samples=MAX_VAL_SAMPLES, blank_idx=blank_idx)

    n_classes = len(CHARSET) + 1
    model = build_model(n_classes)
    optimizer = tfa_adamw(LR, WEIGHT_DECAY)
    ctc = CTCLossLayer(blank_index=blank_idx)

    @tf.function
    def train_step(imgs, labels, label_lens):
        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            loss = ctc(labels, logits, label_lens)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(imgs, labels, label_lens):
        logits = model(imgs, training=False)
        return ctc(labels, logits, label_lens)

    steps_per_epoch = int(math.ceil(min(len(train_rows), MAX_TRAIN_SAMPLES) / BATCH_SIZE))
    progress = ProgressETA(steps_per_epoch=steps_per_epoch, total_epochs=EPOCHS)
    best_val_cer = 1e9

    for epoch in range(EPOCHS):
        print(f"\nDevice: CPU | Epoch {epoch+1}/{EPOCHS}")
        progress.on_epoch_begin(epoch)

        total, count, b = 0.0, 0, 0
        for imgs, labels, label_lens, names, raws in train_ds:
            progress.on_train_batch_begin(b)
            loss_val = float(train_step(imgs, labels, label_lens).numpy())
            bs = int(imgs.shape[0])
            total += loss_val * bs
            count += bs
            progress.on_train_batch_end(b, loss=loss_val, bs=bs)
            b += 1
        train_loss = total / max(1, count)
        progress.on_epoch_end(epoch)

        vtotal, vcount = 0.0, 0
        for imgs, labels, label_lens, names, raws in val_ds:
            vloss = float(val_step(imgs, labels, label_lens).numpy())
            bs = int(imgs.shape[0])
            vtotal += vloss * bs; vcount += bs
        val_loss = vtotal / max(1, vcount)

        val_cer, val_exact = eval_one_epoch(model, val_ds, blank_idx, idx_to_char)
        print(f"[{epoch+1:02d}/{EPOCHS}] train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_CER={val_cer:.4f} | val_exact={val_exact*100:.2f}%")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            model.save(KERAS_PATH, include_optimizer=False)
            print(f"  ✓ Saved new best to {KERAS_PATH}")

    export_all(model, idx_to_char, blank_idx)
    print("Done.")

if __name__ == "__main__":
    main()
