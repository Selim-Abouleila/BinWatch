# app.py  – version heuristique « sans base »
import os, io, datetime, pathlib, statistics
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

import cv2                # ← opencv-python
import numpy as np

# ----------- config & chemins -----------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMG_DIR  = BASE_DIR / "data" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ----------- utils : features & règle -----------
def extract_basic_features(path: pathlib.Path) -> dict:
    """Retourne width, height, size_kb, avg RGB."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    pixels = list(im.getdata())
    avg_r = statistics.mean(p[0] for p in pixels)
    avg_g = statistics.mean(p[1] for p in pixels)
    avg_b = statistics.mean(p[2] for p in pixels)
    size_kb = round(path.stat().st_size / 1024, 2)
    return {
        "filename": path.name,
        "width": w,
        "height": h,
        "size_kb": size_kb,
        "avg_r": round(avg_r, 1),
        "avg_g": round(avg_g, 1),
        "avg_b": round(avg_b, 1),
    }

def ground_trash_ratio(pil_img: Image.Image, crop=0.4) -> float:
    """
    % de pixels 'clairs / plastiques' dans la partie basse de l'image.
    - crop = 0.4 => on examine les 60 % inférieurs (1 - crop).
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h = img.shape[0]
    roi = img[int(h * crop):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # pixels quasi blancs OU très saturés + suffisamment clairs
    mask1 = cv2.inRange(hsv, (0,   0, 180), (180,  40, 255))   # blanc/gris clair
    mask2 = cv2.inRange(hsv, (0,  80, 100), (180, 255, 255))   # plastiques colorés
    mask  = cv2.bitwise_or(mask1, mask2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ratio = mask.sum() / 255 / mask.size
    return round(ratio, 4)

def auto_rule(feat: dict, pil_img: Image.Image) -> str:
    """
    Heuristique : si >12 % de pixels 'clairs' au sol → pleine.
    Sinon vide.
    """
    ratio = ground_trash_ratio(pil_img)
    feat["ground_ratio"] = ratio  # on garde pour debug/front
    return "pleine" if ratio > 0.12 else "vide"

# ----------- routes -----------
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400

    img_file = request.files["image"]
    ts   = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = f"{ts}_{img_file.filename}"
    path = IMG_DIR / name
    img_file.save(path)

    pil   = Image.open(path).convert("RGB")
    feat  = extract_basic_features(path)
    feat["label_auto"] = auto_rule(feat, pil)

    return jsonify(success=True,
                   image_url=f"/images/{name}",
                   features=feat)

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400
    img_bytes = request.files["image"].read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # On écrit quand même sur disque pour réutiliser extract_basic_features
    tmp_path = IMG_DIR / "tmp_upload"
    pil.save(tmp_path, format="JPEG")

    feat  = extract_basic_features(tmp_path)
    label = auto_rule(feat, pil)
    tmp_path.unlink(missing_ok=True)

    return jsonify(success=True, label=label, features=feat)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)

# ----------- main -----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="::", port=port, debug=True)
