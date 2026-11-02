import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hides ALL logs (Info, Warnings, Errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: Turns off the oneDNN notice

import sys
import importlib

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostRegressor
from dotenv import load_dotenv


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))
MODEL_DIR = {
    "recommendation": os.path.join(
        PROJECT_DIR, "crop_recommendation", "model", "crop_recommendation_model.pkl"
    ),
    "price": os.path.join(
        PROJECT_DIR, "crop_price_prediction", "model", "crop_price_prediction_model.cbm"
    ),
    "disease": os.path.join(
        PROJECT_DIR, "crop_disease_detection", "model", "crop_disease_cnn_model.keras"
    ),
}
DISEASE_DATASET_ROOT = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "kagglehub",
    "datasets",
    "vipoooool",
    "new-plant-diseases-dataset",
    "versions",
)

RECOMMENDATION_FIELDS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
PRICE_FIELDS = [
    "STATE",
    "District Name",
    "Market Name",
    "Commodity",
    "Variety",
    "Grade",
    "Min_Price",
    "Max_Price",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
]


DEFAULT_DISEASE_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_disease_labels():
    if os.path.isdir(DISEASE_DATASET_ROOT):
        versions = [
            os.path.join(DISEASE_DATASET_ROOT, name)
            for name in os.listdir(DISEASE_DATASET_ROOT)
            if os.path.isdir(os.path.join(DISEASE_DATASET_ROOT, name))
        ]
        if versions:
            dataset_path = max(versions, key=os.path.getmtime)
            train_dirs = []
            for root, dirs, _ in os.walk(dataset_path):
                if "train" in dirs:
                    train_dirs.append(os.path.join(root, "train"))
            if train_dirs:
                train_dir = train_dirs[0]
                return sorted(
                    name
                    for name in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, name))
                )
    return DEFAULT_DISEASE_LABELS


def classify_disease(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(MODEL_DIR["disease"]):
        raise FileNotFoundError(
            f"Disease detection model not found at '{MODEL_DIR['disease']}'. "
            "Please train or place 'crop_disease_cnn_model.keras' in 'crop_disease_detection/model/'."
        )
    model = tf.keras.models.load_model(MODEL_DIR["disease"])
    image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    predictions = model.predict(np.array([image_array]), verbose=0)[0]
    labels = load_disease_labels()
    index = int(np.argmax(predictions))
    return labels[index], float(predictions[index]) * 100


def predict_recommendation(fields):
    model = joblib.load(MODEL_DIR["recommendation"])
    values = [[float(fields[name]) for name in RECOMMENDATION_FIELDS]]
    frame = pd.DataFrame(values, columns=RECOMMENDATION_FIELDS)
    return str(model.predict(frame)[0])


def predict_price(fields):
    model = CatBoostRegressor()
    model.load_model(MODEL_DIR["price"])
    categorical = PRICE_FIELDS[:6]
    numerical = PRICE_FIELDS[6:]
    values = {name: fields[name] for name in PRICE_FIELDS}
    frame = pd.DataFrame(
        [{name: str(values[name]) if name in categorical else float(values[name])
          for name in categorical + numerical}],
        columns=categorical + numerical,
    )
    return float(model.predict(frame)[0])


def groq_json(client, messages):
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def create_groq_client(api_key):
    """Load the SDK without resolving this chatbot file as the SDK module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_module = sys.modules.get("groq")
    if local_module is not None and getattr(local_module, "__file__", None) == __file__:
        del sys.modules["groq"]
    original_path = list(sys.path)
    sys.path = [path for path in sys.path if os.path.abspath(path or os.curdir) != script_dir]
    try:
        groq_sdk = importlib.import_module("groq")
    finally:
        sys.path = original_path
    return groq_sdk.Groq(api_key=api_key)


def analyze_message(client, message, state):
    schema = {
        "intent": "recommendation | price | disease | general",
        "fields": "object containing only clearly provided values",
        "image_path": "local image path if clearly provided, otherwise null",
    }
    prompt = (
        "Classify the farmer's message and extract only explicit values. "
        "Use recommendation for soil/crop suggestions, price for market-price "
        "prediction, disease for plant disease/image diagnosis, and general for "
        "other farming questions. Never invent missing values. "
        f"Known conversation state: {json.dumps(state)}. "
        f"Return valid JSON matching this shape: {json.dumps(schema)}"
    )
    return groq_json(
        client,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
    )


def missing_fields(fields, required):
    return [name for name in required if name not in fields or fields[name] in ("", None)]


def chat_response(client, message, state, history, image_path=None):
    analysis = analyze_message(client, message, state)
    intent = analysis.get("intent", "general")
    state.update(analysis.get("fields") or {})
    image_path = image_path or analysis.get("image_path")

    if intent == "recommendation":
        missing = missing_fields(state, RECOMMENDATION_FIELDS)
        if missing:
            return f"Please provide these values: {', '.join(missing)}."
        crop = predict_recommendation(state)
        state.clear()
        return f"Based on your soil and weather values, grow {crop}."

    if intent == "price":
        missing = missing_fields(state, PRICE_FIELDS)
        if missing:
            return f"To predict the modal price, please provide: {', '.join(missing)}."
        price = predict_price(state)
        state.clear()
        return f"The predicted modal price is {price:.2f}."

    if intent == "disease":
        if not image_path:
            return "Please attach a plant image so I can run the disease detection model."
        disease, confidence = classify_disease(image_path)
        state.clear()
        return f"Prediction: {disease} ({confidence:.2f}% confidence)."

    history.extend([{"role": "user", "content": message}])
    answer = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Krishi AI, a concise and friendly farming assistant. "
                    "Give practical, safety-conscious advice. Do not claim to have "
                    "run a local model unless the application reports its result."
                ),
            },
            *history[-8:],
        ],
        temperature=0.4,
    ).choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in PowerShell with "
            "$env:GROQ_API_KEY='your-key' before starting the chatbot."
        )

    client = create_groq_client(api_key)
    state = {}
    history = []
    print("Krishi AI chatbot started. Type 'exit' to quit.")
    print("You can ask about crop recommendations, prices, or upload a disease image.")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        try:
            print(f"Bot: {chat_response(client, message, state, history)}")
        except (FileNotFoundError, OSError, ValueError, TypeError) as error:
            print(f"Bot: I could not complete that request: {error}")


if __name__ == "__main__":
    main()