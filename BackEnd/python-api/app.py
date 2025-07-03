# app.py – version auto-apprenante avec heuristique améliorée

import os, io, datetime, pathlib, json
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageStat
import numpy as np, cv2                   # opencv-python-headless
from sklearn.metrics import accuracy_score

BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
IMG_DIR  = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

FEAT_PATH = DATA_DIR / "features.json"
SEUIL_PATH = DATA_DIR / "seuils.json"

app = Flask(__name__)

# ➖➖➖ INIT SEUILS ➖➖➖
SEUILS_DEFAULTS = {
    "taille_ko": 319.4,
    "ground_ratio": 0.23,
    "entropy": 5048.8,
    "contrast": 75.33,
    "dark_pixel_ratio": 0.31
}
SEUILS = SEUILS_DEFAULTS.copy()
if SEUIL_PATH.exists():
    SEUILS.update(json.load(SEUIL_PATH.open()))

# ➖➖➖ UTILS ➖➖➖
def load_resized(stream: bytes, max_side=1024):
    pil = Image.open(io.BytesIO(stream)).convert("RGB")
    pil.thumbnail((max_side, max_side))
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, arr

def get_contrast(img):
    stat = ImageStat.Stat(img)
    return sum(stat.stddev) / 3

def basic_features(arr_bgr, size_bytes, name):
    h, w = arr_bgr.shape[:2]
    b, g, r = cv2.mean(arr_bgr)[:3]
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    entropy_val = float(cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().var())
    pil = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
    arr_gray = np.array(pil.convert("L"))
    dark_ratio = np.sum(arr_gray < 80) / arr_gray.size
    contrast_val = get_contrast(pil)

    return dict(
        filename=name, width=w, height=h,
        size_kb=round(size_bytes/1024, 2),
        avg_r=round(r, 1), avg_g=round(g, 1), avg_b=round(b, 1),
        entropy=round(entropy_val, 2),
        contrast=round(contrast_val, 2),
        dark_pixel_ratio=round(dark_ratio, 3)
    )

# ➖➖➖ DETECTION ➖➖➖
def plast_mask_ratio(arr_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,   0, 200), (180, 40, 255))
    vivid = cv2.inRange(hsv, (0,  80, 130), (180,255,255))
    brown = cv2.inRange(hsv, (5,  40,  40), (25,210,255))
    black = cv2.inRange(hsv, (0,   0,   0), (180, 50, 60))
    mask = white | vivid | brown | black
    soil = cv2.inRange(hsv, (15,  0, 40), (100, 80,255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(soil))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    return mask.sum() / 255 / mask.size

def auto_rule(feat: dict, arr_bgr: np.ndarray, seuils=None) -> str:
    if seuils is None:
        seuils = SEUILS

    h = arr_bgr.shape[0]
    ground_slice = arr_bgr[int(h * 0.55):, :]
    gr = plast_mask_ratio(ground_slice)
    feat["ground_ratio"] = round(gr, 4)

    score = 0
    if feat["size_kb"] > seuils["taille_ko"]: score += 1
    if feat["ground_ratio"] > seuils["ground_ratio"]: score += 1
    if feat["entropy"] > seuils["entropy"]: score += 1
    if feat["contrast"] < seuils["contrast"]: score += 1
    if feat["dark_pixel_ratio"] > seuils["dark_pixel_ratio"]: score += 1

    return "pleine" if score >= 3 else "vide"

# ➖➖➖ DATA ➖➖➖
def save_feature_record(feat):
    all_feats = []
    if FEAT_PATH.exists():
        all_feats = json.load(FEAT_PATH.open())
    all_feats.append(feat)
    json.dump(all_feats, FEAT_PATH.open("w"), indent=2)

def reoptimise_thresholds():
    if not FEAT_PATH.exists(): return
    data = json.load(FEAT_PATH.open())
    seuils_taille   = [200, 300, 400, 500]
    seuils_gr       = [0.1, 0.2, 0.25, 0.3]
    seuils_entropy  = [4000, 4500, 5000, 5500, 6000]
    seuils_contrast = [40, 50, 60, 70, 80]
    seuils_dark     = [0.2, 0.25, 0.3, 0.35, 0.4]

    best_score, best = 0, SEUILS
    for t in seuils_taille:
        for d in seuils_gr:
            for e in seuils_entropy:
                for c in seuils_contrast:
                    for dk in seuils_dark:
                        y_pred = []
                        for f in data:
                            score = 0
                            if f["size_kb"] > t: score += 1
                            if f.get("ground_ratio", 0) > d: score += 1
                            if f.get("entropy", 0) > e: score += 1
                            if f.get("contrast", 100) < c: score += 1
                            if f.get("dark_pixel_ratio", 0) > dk: score += 1
                            y_pred.append("pleine" if score >= 3 else "vide")

                        y_true = [f["label_auto"] for f in data]
                        score_val = accuracy_score(y_true, y_pred)
                        if score_val > best_score:
                            best_score = score_val
                            best = {
                                "taille_ko": t,
                                "ground_ratio": d,
                                "entropy": e,
                                "contrast": c,
                                "dark_pixel_ratio": dk
                            }

    json.dump(best, SEUIL_PATH.open("w"), indent=2)
    SEUILS.update(best)

# ➖➖➖ COEUR ➖➖➖
def process(stream: bytes, orig_name: str, seuils=None):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = f"{ts}_{orig_name}"
    path = IMG_DIR / name
    path.write_bytes(stream)

    pil, arr = load_resized(stream)
    feat = basic_features(arr, path.stat().st_size, name)
    feat["label_auto"] = auto_rule(feat, arr, seuils)
    save_feature_record(feat)
    return name, feat

# ➖➖➖ ROUTES ➖➖➖
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

    seuils = request.form.get("seuils")
    seuils = json.loads(seuils) if seuils else None

    _, feat = process(request.files["image"].read(), "tmp.jpg", seuils)
    return jsonify(success=True, label=feat["label_auto"], features=feat)

@app.route("/images/<path:fname>")
def serve(fname):
    return send_from_directory(IMG_DIR, fname)
@app.route("/api/seuils", methods=["GET"])
def get_seuils():
    return jsonify(SEUILS)

@app.route("/api/seuils", methods=["POST"])
def update_seuils():
    data = request.get_json()
    if not data:
        return jsonify(success=False, error="Aucune donnée reçue"), 400

    # Ne pas modifier SEUILS global, juste renvoyer les seuils reçus
    return jsonify(success=True, seuils=data)

@app.route("/api/seuils/reset", methods=["POST"])
def reset_seuils():
    SEUILS.clear()
    SEUILS.update(SEUILS_DEFAULTS)
    json.dump(SEUILS, SEUIL_PATH.open("w"), indent=2)
    return jsonify(success=True, seuils=SEUILS)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="::", port=port, debug=True)
