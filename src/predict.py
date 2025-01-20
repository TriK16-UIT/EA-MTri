import torch
import os
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration
)
from datasets import Dataset
import transformers
transformers.utils.logging.set_verbosity_error()

from config import REFERENCES_PATH, PREDICTION_PATH, LOCALE_MAP
from utils.utils import load_json, save_json
from tqdm import tqdm

#SETTINGS
MODEL = 'E:/EA-MTri/results_mt5small_mt/checkpoint-74520'
MODEL_NAME = "mt5-small-noarabic"
MAX_LENGTH = 256

#Create path
PREDICTION_VAL_PATH= os.path.join(PREDICTION_PATH, MODEL_NAME, "validation")
os.makedirs(PREDICTION_VAL_PATH, exist_ok=True)
RAW_VAL_PATH = os.path.join(REFERENCES_PATH, "validation")
os.makedirs(RAW_VAL_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

model = MT5ForConditionalGeneration.from_pretrained(MODEL)
tokenizer = T5Tokenizer.from_pretrained(MODEL, model_max_length=MAX_LENGTH)
model.to(DEVICE)

def translate_text(source_text, source_lang, target_lang):
    task_prefix = f"entity translate {source_lang} to {target_lang}: "
    input_text = task_prefix + source_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_LENGTH, padding='max_length', truncation=True).to(DEVICE)

    outputs = model.generate(**inputs, max_length=MAX_LENGTH)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction

for filename in tqdm(os.listdir(RAW_VAL_PATH)):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(RAW_VAL_PATH, filename)
        output_file = os.path.join(PREDICTION_VAL_PATH, filename)

        data = load_json(input_file)

        output_data = []
        for entry in data:
            source_text = entry["source"]
            source_lang = LOCALE_MAP.get(entry["source_locale"])
            target_lang = LOCALE_MAP.get(entry["target_locale"])
            id_ = entry["id"]

            prediction = translate_text(source_text, source_lang, target_lang)

            output_entry = {
                "id": id_,
                "source_language": source_lang,
                "target_language": target_lang,
                "text": source_text,
                "prediction": prediction,
            }
            output_data.append(output_entry)

        save_json(output_data, output_file)