import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from native_language_nlp.groq import (
    RECOMMENDATION_FIELDS,
    PRICE_FIELDS,
    classify_disease,
    predict_recommendation,
    predict_price,
    missing_fields,
)

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DEFAULT_LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_local_model_and_tokenizer():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None or _LOCAL_TOKENIZER is None:
        model_path = os.getenv("LOCAL_NLP_MODEL")
        if not model_path:
            model_path = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else DEFAULT_LOCAL_MODEL_NAME

        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="cpu",
        )
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def extract_regex_fields(text):
    """Fallback entity extraction from text for crop and price parameters."""
    fields = {}
    patterns = {
        "N": r"\bN\s*[:=]\s*([\d\.]+)",
        "P": r"\bP\s*[:=]\s*([\d\.]+)",
        "K": r"\bK\s*[:=]\s*([\d\.]+)",
        "temperature": r"\btemperature\s*[:=]\s*([\d\.]+)",
        "humidity": r"\bhumidity\s*[:=]\s*([\d\.]+)",
        "ph": r"\bph\s*[:=]\s*([\d\.]+)",
        "rainfall": r"\brainfall\s*[:=]\s*([\d\.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                fields[key] = float(m.group(1))
            except ValueError:
                pass
    return fields


def local_analyze_message(message, state):
    model, tokenizer = get_local_model_and_tokenizer()

    schema = {
        "intent": "recommendation | price | disease | general",
        "fields": {},
    }
    prompt = (
        "Classify the message into JSON. "
        "Use 'recommendation' if asking what crop to grow or soil values. "
        "Use 'price' for crop modal/mandi price inquiry. "
        "Use 'disease' for plant disease diagnosis or leaf issues. "
        "Use 'general' for general farming advice or greeting. "
        f"Return ONLY JSON matching: {json.dumps(schema)}"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
    out_tokens = outputs[0][len(inputs.input_ids[0]):]
    raw = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    # Extract JSON from output
    intent = "general"
    extracted_fields = {}
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            intent = parsed.get("intent", "general")
            if isinstance(parsed.get("fields"), dict):
                extracted_fields = parsed.get("fields")
    except Exception:
        pass

    # Heuristic fallback if LLM classification missed obvious cues
    lower = message.lower()
    if any(k in lower for k in ["n=", "p=", "k=", "soil", "grow", "fasal", "recommend"]):
        intent = "recommendation"
    elif any(k in lower for k in ["price", "mandi", "bhav", "daam", "rate"]):
        intent = "price"
    elif any(k in lower for k in ["disease", "bimari", "rog", "leaf", "plant"]):
        intent = "disease"

    regex_vals = extract_regex_fields(message)
    extracted_fields.update(regex_vals)

    return {"intent": intent, "fields": extracted_fields}


def local_chat_response(message, state, history, image_path=None):
    analysis = local_analyze_message(message, state)
    intent = analysis.get("intent", "general")
    state.update(analysis.get("fields") or {})

    if intent == "recommendation":
        missing = missing_fields(state, RECOMMENDATION_FIELDS)
        if missing:
            return f"Please provide these values for crop recommendation: {', '.join(missing)}."
        crop = predict_recommendation(state)
        state.clear()
        return f"Based on your soil and weather values, you should grow {crop}."

    if intent == "price":
        missing = missing_fields(state, PRICE_FIELDS)
        if missing:
            return f"To predict the modal price, please provide: {', '.join(missing)}."
        price = predict_price(state)
        state.clear()
        return f"The predicted modal price is {price:.2f}."

    if intent == "disease":
        if not image_path:
            return "Please attach a plant leaf image so I can run the disease detection model."
        disease, confidence = classify_disease(image_path)
        state.clear()
        return f"Prediction: {disease} ({confidence:.2f}% confidence)."

    # General conversation in farmer's native language using local model
    model, tokenizer = get_local_model_and_tokenizer()
    sys_prompt = (
        "You are Krishi AI, a helpful and knowledgeable agricultural assistant. "
        "Answer the farmer's question concisely, practically, and politely in the same language they use (Hindi, Hinglish, or English)."
    )
    conv_messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-6:]:
        conv_messages.append(turn)
    conv_messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(conv_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.6,
            do_sample=True,
            top_p=0.9,
        )
    out_tokens = outputs[0][len(inputs.input_ids[0]):]
    reply = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return reply
