from pathlib import Path
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

p_fp32 = Path("checkpoints/ctc_ocr_fp32.tflite")
p_fp16 = Path("checkpoints/ctc_ocr_fp16.tflite")

for p in [p_fp32, p_fp16]:
    if not p.exists():
        print("Missing:", p); continue
    b = p.read_bytes()
    ident = b[4:8]            # <-- correct place for 'TFL3'
    print(f"{p.name}: size={len(b)} bytes | ident={ident!r}")
    interp = tflite.Interpreter(model_path=str(p))
    interp.allocate_tensors()
    print(f"  allocate_tensors(): OK")
