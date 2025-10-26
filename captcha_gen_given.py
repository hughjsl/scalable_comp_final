

import csv
import random
from pathlib import Path

from captcha.image import ImageCaptcha

N_SAMPLES = 64000
IMG_W, IMG_H = 192, 96

CHARSET = "123456789adeghjknoswxBCFMPQRTUVYZ=#{}[]%+\\"
MIN_CHARS = 1
MAX_CHARS = 6

FONT_TTF = "fonts/WildCrazy.ttf"
FONT_OTF = "fonts/Ring of Kerry.otf"

OUT_DIR = Path("output_captchas")
FILE_PATTERN = "oldgen_{:06d}.png"

LABELS_CSV  = OUT_DIR / "labels.csv"
ANSWERS_CSV = OUT_DIR / "answers.csv"

GLOBAL_SEED = None



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_text():
    length = random.randint(MIN_CHARS, MAX_CHARS)
    return "".join(random.choice(CHARSET) for _ in range(length))

def build_captcha_generator(font_path):
    if font_path and Path(font_path).exists():
        return ImageCaptcha(width=IMG_W, height=IMG_H, fonts=[font_path])
    return ImageCaptcha(width=IMG_W, height=IMG_H)



def main():
    if GLOBAL_SEED is not None:
        random.seed(GLOBAL_SEED)

    ensure_dir(OUT_DIR)

    fonts = []
    if Path(FONT_TTF).exists(): fonts.append(FONT_TTF)
    if Path(FONT_OTF).exists(): fonts.append(FONT_OTF)
    if not fonts:
        fonts = [None]

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f_labels, \
         ANSWERS_CSV.open("w", newline="", encoding="utf-8") as f_answers:

        labels_writer  = csv.writer(f_labels)
        answers_writer = csv.writer(f_answers)

        labels_writer.writerow(["filename", "label"])

        for i in range(N_SAMPLES):
            text = pick_text()
            font_path = random.choice(fonts)
            gen = build_captcha_generator(font_path)

            fname = FILE_PATTERN.format(i)
            gen.write(text, str(OUT_DIR / fname))

            row = [fname, text]
            labels_writer.writerow(row)
            answers_writer.writerow(row)

            if i % 200 == 0 or i == N_SAMPLES - 1:
                print(f"Generated {i+1}/{N_SAMPLES}")

    print("\nDone.")
    print("Images:", OUT_DIR)
    print("labels.csv:", LABELS_CSV)
    print("answers.csv:", ANSWERS_CSV)

if __name__ == "__main__":
    main()
