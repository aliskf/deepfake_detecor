import os
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
print(f"PyMongo version: {__import__('pymongo').__version__}")
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# --- MongoDB Configuration ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "deepfake_detector"
COLLECTION_NAME = "predictions"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    app.logger.info("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # You might want to handle this more gracefully,
    # maybe by disabling the DB features or exiting the app.
    client = None
    collection = None


# --- Model Loading ---
MODEL_PATH = 'deepfake_detector_model.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

def prepare_image(image, target_size):
    """Prepares image for model prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and database logging."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file and model:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = prepare_image(image, target_size=(128, 128))

            prediction = model.predict(processed_image)[0][0]
            is_deepfake = bool(prediction <= 0.5)
            confidence = float(prediction)

            response = {
                "prediction": "fake" if is_deepfake else "real",
                "confidence": confidence,
                "ref_id": "not_logged"  # Placeholder if MongoDB is not active
            }

            if collection:
                try:
                    prediction_id = str(uuid.uuid4())
                    log_entry = {
                        "id": prediction_id,
                        "timestamp": datetime.now().isoformat(),
                        "filename": file.filename,
                        "prediction": response["prediction"],
                        "confidence": response["confidence"]
                    }
                    collection.insert_one(log_entry)
                    app.logger.info(f"Prediction logged: {prediction_id}")
                    response["ref_id"] = prediction_id
                except Exception as db_e:
                    app.logger.error(f"Error logging to MongoDB: {db_e}")

            return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in simplified predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True)
