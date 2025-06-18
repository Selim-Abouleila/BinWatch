import os, io
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    """
    Receive raw image bytes, run your ML model,
    return a fake label for now.
    """
    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes))

    # TODO: your model here -----------------
    width, height = img.size
    label = "plastic" if width > height else "glass"
    # ---------------------------------------

    return jsonify({"label": label})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))      # Railway injects $PORT
    app.run(host="::", port=port, debug=True)
