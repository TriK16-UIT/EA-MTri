import os
from src.utils.filters import apply_all_filterings
from src.utils.utils import load_json, save_json
from tqdm import tqdm


def process_language_data(data):
    data = apply_all_filterings(data)
    return data

def process_all_languages(input_folder, output_folder):
    for lang_folder in tqdm(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, lang_folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.jsonl'):
                    json_input_path = os.path.join(folder_path, file_name)
                    json_output_path = os.path.join(output_folder, lang_folder, file_name)
                    if not os.path.exists(json_output_path):
                        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

                    raw_data = load_json(json_input_path)
                    preprocessed_data = process_language_data(raw_data)
                    save_json(preprocessed_data, json_output_path)   
    print("All language folders processed successfully!")