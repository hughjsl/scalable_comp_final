
from pathlib import Path
import tensorflow as tf

KERAS_DIR      = Path("checkpoints/ctc_ocr_keras")
META_JSON      = Path("checkpoints/charset.json")


SAVEDMODEL_DIR = Path("checkpoints/savedmodel_ctc_static")
OUT_DIR        = Path("checkpoints")
TFLITE_FP32    = OUT_DIR / "ctc_ocr_fp32.tflite"
TFLITE_FP16    = OUT_DIR / "ctc_ocr_fp16.tflite"

IMG_H, IMG_W = 96, 192
BATCH = 1

def make_concrete(model):

    @tf.function(input_signature=[tf.TensorSpec([BATCH, IMG_H, IMG_W, 1], tf.float32, name="image")])
    def serve(x):
        y = model(x, training=False)
        return {"logits": y}
    return serve.get_concrete_function()

def save_static_savedmodel(model, concrete_fn):
    if SAVEDMODEL_DIR.exists():
        import shutil; shutil.rmtree(SAVEDMODEL_DIR)

    tf.saved_model.save(
        obj=model,
        export_dir=str(SAVEDMODEL_DIR),
        signatures={"serving_default": concrete_fn},
    )
    print("Wrote static SavedModel →", SAVEDMODEL_DIR)

def _convert(use_fp16=False, allow_select_ops=False, legacy=False, from_concrete=False, concrete_fn=None):
    if from_concrete:
        assert concrete_fn is not None
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))

    if legacy:
        converter.experimental_new_converter = False

    if use_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    if allow_select_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

    return converter.convert()

def try_routes(fp16, concrete_fn):
    trials = [
        ("builtins (concrete,new)",   dict(use_fp16=fp16, allow_select_ops=False, legacy=False, from_concrete=True,  concrete_fn=concrete_fn)),
        ("builtins (savedmodel,new)", dict(use_fp16=fp16, allow_select_ops=False, legacy=False, from_concrete=False, concrete_fn=None)),
        ("builtins (concrete,old)",   dict(use_fp16=fp16, allow_select_ops=False, legacy=True,  from_concrete=True,  concrete_fn=concrete_fn)),
        ("builtins (savedmodel,old)", dict(use_fp16=fp16, allow_select_ops=False, legacy=True,  from_concrete=False, concrete_fn=None)),
        ("SELECT_TF_OPS (concrete)",  dict(use_fp16=fp16, allow_select_ops=True,  legacy=False, from_concrete=True,  concrete_fn=concrete_fn)),
        ("SELECT_TF_OPS (savedmodel)",dict(use_fp16=fp16, allow_select_ops=True,  legacy=False, from_concrete=False, concrete_fn=None)),
    ]
    for name, kwargs in trials:
        try:
            fb = _convert(**kwargs)
            print("✓ Convert OK via:", name)
            return fb, ("SELECT_TF_OPS" in name)
        except Exception as e:
            print("× Convert failed via:", name, "|", (repr(e)[:220] + "..."))
    raise RuntimeError("All conversion attempts failed.")

def write_fb(path: Path, flatbuf: bytes) -> bool:
    path.write_bytes(flatbuf)
    ok = flatbuf[:4] == b"TFL3"
    print(f"Wrote {path} ({len(flatbuf)} bytes) | header={flatbuf[:4]!r} | valid={ok}")
    return ok

def main():
    print("TensorFlow:", tf.__version__)
    assert KERAS_DIR.exists(), f"Missing Keras model dir: {KERAS_DIR}"
    assert META_JSON.exists(), f"Missing charset meta: {META_JSON}"

    model = tf.keras.models.load_model(KERAS_DIR, compile=False)
    concrete = make_concrete(model)
    save_static_savedmodel(model, concrete)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # FP32
    fb32, sel32 = try_routes(fp16=False, concrete_fn=concrete)
    ok32 = write_fb(TFLITE_FP32, fb32)
    print("FP32 uses SELECT_TF_OPS:", sel32)

    # FP16
    fb16, sel16 = try_routes(fp16=True, concrete_fn=concrete)
    ok16 = write_fb(TFLITE_FP16, fb16)
    print("FP16 uses SELECT_TF_OPS:", sel16)

    if not ok32 or not ok16:
        print("⚠️ One of the flatbuffers failed the header check; re-run/inspect the failing route.")
    if sel32 or sel16:
        print("⚠️ SELECT_TF_OPS present → those models will NOT run with `tflite_runtime` on Raspberry Pi (armv7).")
        print("   They require full TF Lite with Flex; aim for a builtins-only conversion.")

if __name__ == "__main__":
    main()
