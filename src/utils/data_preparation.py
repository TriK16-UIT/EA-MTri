import os
import glob
from utils.filters import apply_all_filterings
from utils.ner_tags import add_ner_tags, initialize_spacy_models, remove_and_replace_tags, remove_non_matching_tags, build_ner_dataset
from utils.instructions import add_task_instructions
from utils.utils import load_json, save_json
from config import REFERENCES_PATH, PREPROCESSED_PATH, DATASET_PATH
from tqdm import tqdm

RAW_TRAIN_PATH = os.path.join(REFERENCES_PATH, "train")
os.makedirs(RAW_TRAIN_PATH, exist_ok=True)

PREPROCESSED_TRAIN_PATH = os.path.join(PREPROCESSED_PATH, "train")
os.makedirs(PREPROCESSED_TRAIN_PATH, exist_ok=True)

def process_language_data(data):
    data = apply_all_filterings(data)
    return data

def process_language_folder(input_folder, output_folder):
    train_file = os.path.join(input_folder, "train.jsonl")
    train_MT_file = os.path.join(output_folder, "train_MT.jsonl")
    train_NER_file = os.path.join(output_folder, "train_NER.jsonl")
    train_EA_MT_file = os.path.join(output_folder, "train_EA_MT.jsonl")
    raw_data = load_json(train_file)
    preprocessed_data = process_language_data(raw_data)
    
    #Translation task - keep plain text in both side
    instructed_plain_data = add_task_instructions(preprocessed_data, task_type="translation")
    save_json(instructed_plain_data, train_MT_file)

    initialize_spacy_models()

    #NER recognization task - add NE tags on the target side
    #Right now, only recognize entities in English (need to modify if make available for all)
    ner_data = build_ner_dataset(preprocessed_data)
    ner_tagged_target_data = add_ner_tags(ner_data, ner_target="target")
    ner_tagged_target_data = remove_and_replace_tags(ner_tagged_target_data, ner_target="target")
    instructed_ner_tagged_target_data = add_task_instructions(ner_tagged_target_data, task_type="NER")
    save_json(instructed_ner_tagged_target_data, train_NER_file)

    #EA-MT task - add NE tags on the source side
    ner_tagged_source_data = add_ner_tags(preprocessed_data, ner_target="source")
    ner_tagged_source_data = remove_and_replace_tags(ner_tagged_source_data, ner_target="source")
    # ner_tagged_both_data = remove_non_matching_tags(ner_tagged_both_data)
    instructed_ner_tagged_source_data = add_task_instructions(ner_tagged_source_data, task_type="EA_MT")
    save_json(instructed_ner_tagged_source_data, train_EA_MT_file)
 
def process_all_languages():
    for lang_folder in tqdm(os.listdir(RAW_TRAIN_PATH)):
        input_folder_path = os.path.join(RAW_TRAIN_PATH, lang_folder)
        output_folder_path = os.path.join(PREPROCESSED_TRAIN_PATH, lang_folder)
        print(output_folder_path)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
        process_language_folder(input_folder_path, output_folder_path)
    print("All language folders processed successfully!")

def build_dataset():
    all_data = []
    for lang_folder in tqdm(os.listdir(PREPROCESSED_TRAIN_PATH)):
        lang_folder_path = os.path.join(PREPROCESSED_TRAIN_PATH, lang_folder)

        for train_file_path in glob.glob(os.path.join(lang_folder_path, "*.jsonl")):
            data = load_json(train_file_path)
            for item in data:
                all_data.append({"src": item["source"], "trg": item["target"]})
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    output_file = os.path.join(DATASET_PATH, "train.jsonl")
    save_json(all_data, output_file)
    print(f"Grouped all JSONL data into {output_file}")
