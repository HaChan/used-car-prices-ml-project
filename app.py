from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    with open('model.bin', 'rb') as f:
        dv, model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found! Ensure model.bin exists")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        features = dv.transform(data)

        predictions = model.predict(features)

        response = {
            "predictions": predictions.tolist(),
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "dv_loaded": dv is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
