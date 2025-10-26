
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from image_processing import PreprocConfig, preprocess_pipeline, load_rgb

REAL_DIR = Path("real_captchas")
N_SAMPLES_DEFAULT = 5

def main():
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES_DEFAULT
    except Exception:
        n = N_SAMPLES_DEFAULT

    if not REAL_DIR.exists():
        raise SystemExit(f"Folder not found: {REAL_DIR.resolve()}")

    files = sorted(list(REAL_DIR.glob("*.png")))
    if not files:
        raise SystemExit(f"No PNGs found in {REAL_DIR.resolve()}")

    n = min(n, len(files))
    random.seed()
    sample = random.sample(files, n)

    cfg = PreprocConfig()

    fig_h = max(3, n * 2.2)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(10, fig_h))
    if n == 1:
        axs = np.array([axs])

    for i, p in enumerate(sample):

        rgb = load_rgb(p)
        _, _, bw_clean, _, removed = preprocess_pipeline(rgb, cfg)

        ax0 = axs[i, 0]
        ax0.imshow(rgb)
        ax0.set_title(f"{p.name}\nOriginal", fontsize=10)
        ax0.axis("off")


        ax1 = axs[i, 1]
        ax1.imshow(bw_clean, cmap="gray")
        ax1.set_title(f"Processed (removed dots: {removed})", fontsize=10)
        ax1.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
