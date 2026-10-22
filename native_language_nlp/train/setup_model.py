import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "model")

print(f"Downloading and saving Native Language NLP model ({MODEL_NAME})...")
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Tokenizer saved to {SAVE_DIR}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_DIR)
print(f"Model weights saved to {SAVE_DIR}")
print("Native Language NLP model setup complete!")
