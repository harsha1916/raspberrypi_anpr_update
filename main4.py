#!/usr/bin/env python3
import os, time, re
from pathlib import Path

import cv2
import easyocr
import numpy as np

# ========== EDIT ME ==========
IMAGE_PATH = "images/1.jpg"   # <- put your image here
SAVE_PATH  = ""  # leave "" to auto-generate alongside input, or set "/home/pi5/out.jpg"
LANGS = ['en']   # languages for EasyOCR; add 'en' only for plates
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # for license plates
# =============================

def preprocess(img_bgr, max_side=1600):
    """Lightweight enhancement: resize (keeps aspect), denoise, CLAHE."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), cv2.INTER_LINEAR)

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 30, 30)  # preserve edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    # keep 3ch for drawing
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def draw_result(img, bbox, text, conf, color=(0, 255, 0)):
    # bbox is 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    x, y = pts[0]
    label = f"{text} ({conf:.2f})"
    y = max(0, y - 6)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def normalize_plate(s: str) -> str:
    """Simple cleanup for plates like AP40M0116 / AP40 0LL6."""
    s = re.sub(r"[^A-Za-z0-9]", "", s.upper())
    # Fix common confusions
    trans = str.maketrans({"Ø":"0","O":"0","D":"0","Q":"0","I":"1","L":"1","|":"1","!":"1","S":"5","B":"8"})
    s = s.translate(trans)
    return s

def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print("[INFO] Loading EasyOCR reader (CPU)…")
    t0 = time.time()
    reader = easyocr.Reader(LANGS, gpu=False, verbose=False)
    print(f"[INFO] Reader ready in {time.time()-t0:.2f}s")

    print(f"[INFO] Reading image: {img_path}")
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("cv2.imread returned None (unsupported file or bad path).")

    img_prep = preprocess(img_bgr)

    # Run OCR (detail=1 returns (bbox, text, conf))
    t1 = time.time()
    results = reader.readtext(
        img_prep,
        detail=1,
        paragraph=False,
        allowlist=ALLOWLIST  # bias toward plate characters
    )
    dt = time.time() - t1
    print(f"[INFO] OCR done in {dt:.2f}s, found {len(results)} regions")

    # Draw & print
    out = img_bgr.copy()
    best_plate, best_conf = "", 0.0
    for (bbox, text, conf) in results:
        clean = normalize_plate(text)
        draw_result(out, bbox, clean, conf)
        print(f"  - '{clean}'  conf={conf:.3f}")
        if conf > best_conf and len(clean) >= 6:
            best_conf, best_plate = conf, clean

    if best_plate:
        print(f"[RESULT] Best plate: {best_plate} (conf={best_conf:.3f})")
    else:
        print("[RESULT] No strong plate-like text found.")

    # Save
    if SAVE_PATH:
        save_path = Path(SAVE_PATH)
    else:
        save_path = img_path.with_name(img_path.stem + "_ocr" + img_path.suffix)
    cv2.imwrite(str(save_path), out)
    print(f"[INFO] Saved annotated image to: {save_path}")

if __name__ == "__main__":
    main()
