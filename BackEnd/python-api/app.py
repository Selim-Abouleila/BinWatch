# app.py – heuristique « trash au sol » corrigée

import os, io, datetime, pathlib
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np, cv2                   # opencv-python-headless

BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMG_DIR  = BASE_DIR / "data" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ───────── utils ─────────
def load_resized(stream: bytes, max_side=1024):
    pil = Image.open(io.BytesIO(stream)).convert("RGB")
    pil.thumbnail((max_side, max_side))
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, arr

def basic_features(arr_bgr, size_bytes, name):
    h, w = arr_bgr.shape[:2]
    b, g, r = cv2.mean(arr_bgr)[:3]
    return dict(filename=name, width=w, height=h,
                size_kb=round(size_bytes/1024, 2),
                avg_r=round(r, 1), avg_g=round(g, 1), avg_b=round(b, 1))

# ───────── détection des pixels « déchets » ─────────
def plast_mask_ratio(arr_bgr: np.ndarray) -> float:
    """
    Renvoie la fraction de pixels correspondant à sacs plastiques, papiers,
    cartons, sacs noirs brillants, après avoir éliminé herbe/terre/chaussée.
    """
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0,   0, 200), (180, 40, 255))     # blanc/gris clair
    vivid = cv2.inRange(hsv, (0,  80, 130), (180,255,255))      # couleurs saturées
    brown = cv2.inRange(hsv, (5,  40,  40), (25,210,255))       # carton brun
    black = cv2.inRange(hsv, (0,   0,   0), (180, 50, 60))      # sacs noirs/luisants

    mask = white | vivid | brown | black

    # élimine herbe/terre/chaussée (H 15–100, S < 80, V > 40)
    soil = cv2.inRange(hsv, (15,  0, 40), (100, 80,255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(soil))

    # nettoyage
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    return mask.sum() / 255 / mask.size     # valeur 0-1

def auto_rule(feat: dict, arr_bgr: np.ndarray) -> str:
    """
    Pleine si >= 8 % de pixels « déchets » dans la bande basse (45 %)
    """
    h = arr_bgr.shape[0]
    ground_slice = arr_bgr[int(h*0.55): , :]
    gr = plast_mask_ratio(ground_slice)
    feat["ground_ratio"] = round(gr, 4)
    return "pleine" if gr > 0.08 else "vide"

# ───────── cœur commun ─────────
def process(stream: bytes, orig_name: str):
    ts   = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = f"{ts}_{orig_name}"
    path = IMG_DIR / name
    path.write_bytes(stream)

    pil, arr = load_resized(stream)
    feat     = basic_features(arr, path.stat().st_size, name)
    feat["label_auto"] = auto_rule(feat, arr)
    return name, feat

# ───────── routes ─────────
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400
    name, feat = process(request.files["image"].read(),
                         request.files["image"].filename)
    return jsonify(success=True, image_url=f"/images/{name}", features=feat)

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400
    _, feat = process(request.files["image"].read(), "tmp.jpg")
    return jsonify(success=True, label=feat["label_auto"], features=feat)

@app.route("/images/<path:fname>")
def serve(fname):
    return send_from_directory(IMG_DIR, fname)

# ───────── run local ─────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="::", port=port, debug=True)
