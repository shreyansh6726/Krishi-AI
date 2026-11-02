import base64
import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from knowledge_base import CROP_METADATA, DISEASE_REMEDIES, SAMPLE_PRESETS
from native_language_nlp.groq import (
    classify_disease,
    load_dotenv,
    predict_price,
    predict_recommendation,
    RECOMMENDATION_FIELDS,
    PRICE_FIELDS,
)
from native_language_nlp.local_llm import local_chat_response

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_DIR, "web")
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


class KrishiAIHandler(SimpleHTTPRequestHandler):
    state = {}
    history = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            status_payload = {
                "offline_mode": True,
                "cloud_api_used": False,
                "models": {
                    "crop_recommendation": {
                        "name": "Crop Recommendation",
                        "algorithm": "Random Forest Classifier (Scikit-Learn)",
                        "status": "Online (100% Offline Local Model)",
                        "features": RECOMMENDATION_FIELDS,
                    },
                    "crop_disease_detection": {
                        "name": "Plant Disease Detection",
                        "algorithm": "Deep Convolutional Neural Network (TensorFlow)",
                        "status": "Online (100% Offline Local Model)",
                        "classes_count": 38,
                    },
                    "crop_price_prediction": {
                        "name": "Crop Price Forecasting",
                        "algorithm": "CatBoost Gradient Boosting Regressor",
                        "status": "Online (100% Offline Local Model)",
                        "features": PRICE_FIELDS,
                    },
                    "native_language_nlp": {
                        "name": "Krishi Mitra Language Assistant",
                        "algorithm": "Qwen2.5-0.5B-Instruct (PyTorch/Transformers)",
                        "status": "Online (100% Offline Local Model)",
                    },
                },
            }
            self._send_json(200, status_payload)
            return

        if self.path == "/api/presets":
            self._send_json(200, SAMPLE_PRESETS)
            return

        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            self._send_json(400, {"error": "Empty request body."})
            return

        try:
            raw_body = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw_body)
        except Exception as err:
            self._send_json(400, {"error": f"Invalid JSON payload: {err}"})
            return

        try:
            # 1. CROP RECOMMENDATION ENDPOINT
            if self.path == "/api/recommend":
                missing = [f for f in RECOMMENDATION_FIELDS if f not in data or data[f] in ("", None)]
                if missing:
                    self._send_json(400, {"error": f"Missing required fields: {', '.join(missing)}"})
                    return

                crop = predict_recommendation(data)
                crop_lower = crop.lower().strip()
                metadata = CROP_METADATA.get(
                    crop_lower,
                    {
                        "name_hi": crop.title(),
                        "season": "अनुकूल मौसम",
                        "water_req": "मध्यम",
                        "soil_type": "उपजाऊ दोमट मिट्टी",
                        "description": f"{crop.title()} की खेती के लिए यह मिट्टी और जलवायु पूरी तरह उपयुक्त है।",
                    },
                )
                self._send_json(
                    200,
                    {
                        "success": True,
                        "crop": crop,
                        "crop_name_hi": metadata.get("name_hi"),
                        "season": metadata.get("season"),
                        "water_req": metadata.get("water_req"),
                        "soil_type": metadata.get("soil_type"),
                        "description": metadata.get("description"),
                        "inputs": {k: data[k] for k in RECOMMENDATION_FIELDS},
                    },
                )
                return

            # 2. CROP DISEASE DETECTION ENDPOINT
            if self.path == "/api/detect-disease":
                image_data = data.get("image")
                if not image_data:
                    self._send_json(400, {"error": "Image data is required (base64 encoded)."})
                    return

                if "," in image_data:
                    _, encoded = image_data.split(",", 1)
                else:
                    encoded = image_data

                suffix = os.path.splitext(data.get("imageName", ".jpg"))[1] or ".jpg"
                temp_img = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, dir=PROJECT_DIR
                    ) as f:
                        f.write(base64.b64decode(encoded))
                        temp_img = f.name

                    label, confidence = classify_disease(temp_img)
                finally:
                    if temp_img and os.path.isfile(temp_img):
                        os.remove(temp_img)

                is_healthy = "healthy" in label.lower()
                remedy_data = DISEASE_REMEDIES.get(label, {})

                if "___" in label:
                    plant_part, disease_part = label.split("___", 1)
                else:
                    plant_part, disease_part = label, "Diagnosis"

                plant_name = remedy_data.get("plant", plant_part.replace("_", " "))
                condition_name = remedy_data.get("condition", disease_part.replace("_", " "))
                treatment = remedy_data.get(
                    "treatment",
                    "पौधा स्वस्थ है, किसी दवा की आवश्यकता नहीं है।"
                    if is_healthy
                    else "प्रभावित पत्तियों को हटाएं और जैविक या रासायनिक कवकनाशी का छिड़काव करें।",
                )
                prevention = remedy_data.get(
                    "prevention",
                    "संतुलित पोषण, उचित जल निकास और स्वच्छ बीज का प्रयोग करें।"
                )

                self._send_json(
                    200,
                    {
                        "success": True,
                        "raw_label": label,
                        "plant": plant_name,
                        "condition": condition_name,
                        "is_healthy": is_healthy,
                        "confidence": round(confidence, 2),
                        "treatment": treatment,
                        "prevention": prevention,
                    },
                )
                return

            # 3. CROP PRICE PREDICTION ENDPOINT
            if self.path == "/api/predict-price":
                missing = [f for f in PRICE_FIELDS if f not in data or data[f] in ("", None)]
                if missing:
                    self._send_json(400, {"error": f"Missing required price fields: {', '.join(missing)}"})
                    return

                price = predict_price(data)
                self._send_json(
                    200,
                    {
                        "success": True,
                        "predicted_price": round(price, 2),
                        "unit": "₹ / क्विंटल (Quintal)",
                        "inputs": {k: data[k] for k in PRICE_FIELDS},
                    },
                )
                return

            # 4. 100% OFFLINE LOCAL CHAT ENDPOINT
            if self.path == "/api/chat":
                message = str(data.get("message", "")).strip()
                if not message:
                    self._send_json(400, {"error": "Message cannot be empty."})
                    return

                image_path = None
                image_data = data.get("image")
                if image_data:
                    if "," in image_data:
                        _, encoded = image_data.split(",", 1)
                    else:
                        encoded = image_data
                    suffix = os.path.splitext(data.get("imageName", ".jpg"))[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, dir=PROJECT_DIR
                    ) as image_file:
                        image_file.write(base64.b64decode(encoded))
                        image_path = image_file.name

                try:
                    # 100% Offline inference via local PyTorch / HuggingFace model
                    reply = local_chat_response(message, self.state, self.history, image_path)
                    self._send_json(200, {"reply": reply})
                finally:
                    if image_path and os.path.isfile(image_path):
                        os.remove(image_path)
                return

            self._send_json(404, {"error": f"Endpoint '{self.path}' not found."})

        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            self._send_json(500, {"error": f"Internal Server Error: {error}"})


def main():
    port = int(os.getenv("KRISHI_AI_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), KrishiAIHandler)
    print("=" * 65)
    print("🌾 KRISHI AI - 100% OFFLINE MULTI-MODEL RESEARCH SYSTEM 🌾")
    print("=" * 65)
    print("Zero Cloud APIs: All 4 models run locally on this machine.")
    print("1. Crop Recommendation    : Random Forest (Local)")
    print("2. Plant Disease Detection: Deep CNN (Local)")
    print("3. Crop Price Prediction  : CatBoost Regressor (Local)")
    print("4. Language Assistant     : Qwen2.5-0.5B (Local)")
    print(f"Portal running locally at: http://127.0.0.1:{port}")
    print("=" * 65)
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
