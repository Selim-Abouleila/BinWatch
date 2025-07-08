# app.py – version auto-apprenante avec heuristique améliorée

import os, io, datetime, pathlib, json
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageStat
import numpy as np, cv2                   # opencv-python-headless
from sklearn.metrics import accuracy_score

BASE_DIR    = pathlib.Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "data"
IMG_DIR     = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

FEAT_PATH   = DATA_DIR / "features.json"
SEUIL_PATH  = DATA_DIR / "seuils.json"

app = Flask(__name__)

# ➖➖➖ INIT SEUILS ➖➖➖
SEUILS = {
    "taille_ko":         250,
    "ground_ratio":      0.08,
    "entropy":           5.5,
    "contrast":          30.0,
    "dark_pixel_ratio":  0.2
}
if SEUIL_PATH.exists():
    SEUILS.update(json.load(SEUIL_PATH.open()))

# ── FONCTIONS UTILITAIRES ──────────────────────────────────────────
def load_resized(stream: bytes, max_side=1024):
    """Charge une image en mémoire, la réduit si nécessaire."""
    pil = Image.open(io.BytesIO(stream)).convert("RGB")
    pil.thumbnail((max_side, max_side))
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, arr

def get_contrast(img: Image.Image) -> float:
    stat = ImageStat.Stat(img)
    return sum(stat.stddev) / 3

def basic_features(arr_bgr, size_bytes, name):
    """Extrait les caractéristiques de base d’une image."""
    h, w      = arr_bgr.shape[:2]
    size_kb    = round(size_bytes / 1024, 2)
    b, g, r    = cv2.mean(arr_bgr)[:3]
    pil        = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
    entropy_val  = round(ImageStat.Stat(pil).entropy(), 2)
    contrast_val = round(get_contrast(pil), 2)
    gray         = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    dark_ratio   = round(np.sum(gray < 80) / gray.size, 3)
    # ground_ratio à ajouter si nécessaire…
    return {
        "filename":         name,
        "width":            w,
        "height":           h,
        "size_kb":          size_kb,
        "avg_r":            round(r,1),
        "avg_g":            round(g,1),
        "avg_b":            round(b,1),
        "entropy":          entropy_val,
        "contrast":         contrast_val,
        "dark_pixel_ratio": dark_ratio,
        # "ground_ratio":    ground_ratio,
    }

def auto_rule(feat, arr_bgr):
    """Décide “pleine” ou “vide” selon les SEUILS."""
    score = 0
    if feat["size_kb"]            > SEUILS["taille_ko"]:        score += 1
    if feat.get("ground_ratio",0) > SEUILS["ground_ratio"]:     score += 1
    if feat["entropy"]            > SEUILS["entropy"]:           score += 1
    if feat["contrast"]           < SEUILS["contrast"]:          score += 1
    if feat["dark_pixel_ratio"]   > SEUILS["dark_pixel_ratio"]:  score += 1
    return "pleine" if score >= 3 else "vide"

def save_feature_record(feat):
    """Sauvegarde l’historique dans features.json."""
    all_feats = json.load(FEAT_PATH.open()) if FEAT_PATH.exists() else []
    all_feats.append(feat)
    json.dump(all_feats, FEAT_PATH.open("w"), indent=2)

def reoptimise_thresholds():
    """(Optionnel) Réoptimise automatiquement les SEUILS selon l’historique."""
    if not FEAT_PATH.exists():
        return
    data = json.load(FEAT_PATH.open())
    # … votre algorithme de ré-optimisation …
    # json.dump(nouveaux_seuils, SEUIL_PATH.open("w"), indent=2)
    # SEUILS.update(nouveaux_seuils)

def process(stream: bytes, filename: str):
    """Pipeline complet : resize, features, auto-label, save."""
    pil, arr = load_resized(stream)
    size_kb  = round(len(stream) / 1024, 2)
    feat     = basic_features(arr, len(stream), filename)

    # Label automatique
    feat["label_auto"] = auto_rule(feat, arr)

    # Enregistrement historique
    save_feature_record(feat)

    # Sauvegarde de l’image
    path = IMG_DIR / filename
    with open(path, "wb") as f:
        f.write(stream)

    return filename, feat

# ── ROUTES ─────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400
    name, feat = process(request.files["image"].read(),
                         request.files["image"].filename)
    return jsonify(success=True, image_url=f"/images/{name}", features=feat)

@app.route("/seuils", methods=["GET"])
def get_seuils():
    """Renvoie les seuils actuels pour pré-remplir le formulaire."""
    return jsonify(success=True, seuils=SEUILS)

@app.route("/seuils", methods=["POST"])
def set_seuils():
    """Reçoit et enregistre les nouveaux seuils choisis par l’utilisateur."""
    data = request.get_json() or {}
    for k in SEUILS:
        if k in data:
            SEUILS[k] = data[k]
    json.dump(SEUILS, SEUIL_PATH.open("w"), indent=2)
    return jsonify(success=True, seuils=SEUILS)

@app.route("/stats", methods=["GET"])
def get_stats():
    """
    Renvoie pour chaque seuil (taille_ko, entropy, contrast, etc.)
    les stats min/max/moyenne calculées sur features.json.
    """
    if not FEAT_PATH.exists():
        empty = {k: {"min": 0, "max": 0, "mean": 0} for k in SEUILS}
        return jsonify(success=True, stats=empty)
    feats = json.load(FEAT_PATH.open())
    mapping = {
        "taille_ko":        "size_kb",
        "ground_ratio":     "ground_ratio",
        "entropy":          "entropy",
        "contrast":         "contrast",
        "dark_pixel_ratio": "dark_pixel_ratio"
    }
    stats = {}
    for seuil_key, feat_key in mapping.items():
        vals = [f.get(feat_key, 0) for f in feats if feat_key in f]
        if vals:
            stats[seuil_key] = {
                "min":  round(min(vals), 2),
                "max":  round(max(vals), 2),
                "mean": round(sum(vals) / len(vals), 2)
            }
        else:
            stats[seuil_key] = {"min": 0, "max": 0, "mean": 0}
    return jsonify(success=True, stats=stats)

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    if "image" not in request.files:
        return jsonify(success=False, error="Aucun fichier"), 400
    _, feat = process(request.files["image"].read(), "tmp.jpg")
    return jsonify(success=True, label=feat["label_auto"], features=feat)

@app.route("/images/<path:fname>")
def serve(fname):
    return send_from_directory(IMG_DIR, fname)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
