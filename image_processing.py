
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    cv2.setUseOptimized(True)
except Exception:
    pass
try:

    cv2.setNumThreads(1)
except Exception:
    pass


@dataclass(frozen=True)
class PreprocConfig:

    ADAPT_BLOCK: int = 31
    ADAPT_C: int = 7
    ERODE_AFTER_BIN: bool = True   # optional 1px shave
    ERODE_KERNEL: Tuple[int, int] = (2, 2)


    DOT_MAX_W: int = 8
    DOT_MAX_H: int = 8
    DOT_MAX_AREA: int = 49
    DOT_MARGIN: int = 2
    DOT_ALLOW_RING_FG: int = 2


    MIN_W: int = 5
    MIN_H: int = 10


    UNDERLINE_WHR: float = 4.0




def load_rgb(img: Union[str, Path, np.ndarray]) -> np.ndarray:

    if isinstance(img, (str, Path)):
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def binarize_thin(img_rgb: np.ndarray, cfg: PreprocConfig) -> Tuple[np.ndarray, np.ndarray]:

    dark = np.min(img_rgb, axis=2).astype(np.uint8)


    bw = cv2.adaptiveThreshold(
        dark, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=int(cfg.ADAPT_BLOCK) | 1,  # ensure odd
        C=int(cfg.ADAPT_C)
    )

    if cfg.ERODE_AFTER_BIN:
        k = _get_kernel(cfg.ERODE_KERNEL)
        bw = cv2.erode(bw, k, iterations=1)

    return dark, bw


def remove_isolated_dots(bw: np.ndarray, cfg: PreprocConfig) -> Tuple[np.ndarray, int]:

    m = (bw == 255).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    clean = m.copy()
    removed = 0


    integral = cv2.integral(m, sdepth=cv2.CV_32S)

    def sum_rect(x0: int, y0: int, x1: int, y1: int) -> int:

        return int(
            integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        )

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w > cfg.DOT_MAX_W or h > cfg.DOT_MAX_H or area > cfg.DOT_MAX_AREA:
            continue

        x0 = max(0, x - cfg.DOT_MARGIN)
        y0 = max(0, y - cfg.DOT_MARGIN)
        x1 = min(W, x + w + cfg.DOT_MARGIN)
        y1 = min(H, y + h + cfg.DOT_MARGIN)


        comp_sum = sum_rect(x, y, x + w, y + h)

        win_sum = sum_rect(x0, y0, x1, y1)

        ring_fg = win_sum - comp_sum


        if ring_fg <= cfg.DOT_ALLOW_RING_FG:

            mask = (labels[y:y + h, x:x + w] == i)
            clean[y:y + h, x:x + w][mask] = 0
            removed += 1

    return (clean * 255).astype(np.uint8), removed


def find_components(clean_bw: np.ndarray, cfg: PreprocConfig) -> List[Tuple[int, int, int, int]]:

    m = (clean_bw == 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H = m.shape[0]
    boxes: List[Tuple[int, int, int, int]] = []

    for i in range(1, num):  # skip background
        x, y, w, h, area = stats[i]
        if w < cfg.MIN_W or h < cfg.MIN_H:
            continue

        if (w / max(h, 1)) > cfg.UNDERLINE_WHR and y > H // 2:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def preprocess_pipeline(
    img: Union[str, Path, np.ndarray],
    cfg: PreprocConfig = PreprocConfig()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int, int]], int]:

    rgb = load_rgb(img)
    _, bw = binarize_thin(rgb, cfg)
    bw_clean, removed = remove_isolated_dots(bw, cfg)
    boxes = find_components(bw_clean, cfg)
    dark = np.min(rgb, axis=2).astype(np.uint8)
    return rgb, dark, bw_clean, boxes, removed



_kernel_cache: dict[Tuple[int, int], np.ndarray] = {}

def _get_kernel(shape: Tuple[int, int]) -> np.ndarray:
    """Cache small structuring elements to avoid reallocations."""
    if shape not in _kernel_cache:
        _kernel_cache[shape] = np.ones(shape, np.uint8)
    return _kernel_cache[shape]



if __name__ == "__main__":
    demo_folder = Path("output_captchas")
    files = sorted(demo_folder.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNGs found in {demo_folder.resolve()}")

    cfg = PreprocConfig()
    p = files[0]
    rgb, dark, bw_clean, boxes, removed = preprocess_pipeline(p, cfg)
    print(f"Image: {p.name}")
    print(f"Removed isolated dots: {removed}")
    print(f"Boxes ({len(boxes)}): {boxes[:6]}{' ...' if len(boxes) > 6 else ''}")


    out_dir = Path("preproc_debug")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / (p.stem + "_dark.png")), dark)
    cv2.imwrite(str(out_dir / (p.stem + "_bwclean.png")), bw_clean)
