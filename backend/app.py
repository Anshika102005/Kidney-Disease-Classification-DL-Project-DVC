import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from cnnClassifier.utils.load_trained_model import LoadTrainedModel

MODEL_PATH = ROOT_DIR / "artifacts" / "training" / "trained_model.keras"
LABELS = ["Cyst", "Normal", "Stone", "Tumor"]
TARGET_SIZE = (160, 160)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app)

model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = LoadTrainedModel.load(str(MODEL_PATH), compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH.name})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image file"}), 400

    try:
        image = Image.open(file.stream)
        input_data = preprocess_image(image)
        preds = model.predict(input_data, verbose=0)[0]
        index = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100.0)
        prediction = LABELS[index] if index < len(LABELS) else str(index)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "model": MODEL_PATH.name,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
