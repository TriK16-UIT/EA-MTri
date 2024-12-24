import os
import glob
from src.utils.filters import apply_all_filterings
from src.utils.ner_tags import add_ner_tags, initialize_spacy_models, remove_and_replace_tags, remove_non_matching_tags, build_ner_dataset
from src.utils.instructions import add_task_instructions
from src.utils.utils import load_json, save_json
from config import RAW_TRAIN_PATH, PREPROCESSED_TRAIN_PATH, DATASET_PATH
from tqdm import tqdm

def process_language_data(data):
    data = apply_all_filterings(data)
    return data

def process_language_folder(input_folder, output_folder):
    train_file = os.path.join(input_folder, "train.jsonl")
    train_MT_file = os.path.join(output_folder, "train_MT.jsonl")
    train_NER_file = os.path.join(output_folder, "train_NER.jsonl")
    train_EA_MT_file = os.path.join(output_folder, "train_EA_MT.jsonl")
    raw_data = load_json(train_file)
    
    #Translation task
    preprocessed_data = process_language_data(raw_data)
    instructed_plain_data = add_task_instructions(preprocessed_data, task_type="translation")
    save_json(instructed_plain_data, train_MT_file)

    initialize_spacy_models()

    #NER recognization task
    ner_data = build_ner_dataset(preprocessed_data)
    ner_tagged_target_data = add_ner_tags(ner_data, ner_target="target")
    ner_tagged_target_data = remove_and_replace_tags(ner_tagged_target_data, ner_target="target")
    instructed_ner_tagged_target_data = add_task_instructions(ner_tagged_target_data, task_type="NER")
    save_json(instructed_ner_tagged_target_data, train_NER_file)

    #EA-MT task
    ner_tagged_both_data = add_ner_tags(preprocessed_data, ner_target="target")
    ner_tagged_both_data = add_ner_tags(ner_tagged_both_data, ner_target="source")
    ner_tagged_both_data = remove_and_replace_tags(ner_tagged_both_data, ner_target="target")
    ner_tagged_both_data = remove_and_replace_tags(ner_tagged_both_data, ner_target="source")
    ner_tagged_both_data = remove_non_matching_tags(ner_tagged_both_data)
    instructed_ner_tagged_both_data = add_task_instructions(ner_tagged_both_data, task_type="EA_MT")
    save_json(instructed_ner_tagged_both_data, train_EA_MT_file)

    # ner_tagged_source_data = add_ner_tags(preprocessed_data, ner_target="source")
    # instructed_ner_tagged_source_data = add_task_instructions(ner_tagged_source_data, task_type="EA_MT")
    # save_json(instructed_ner_tagged_source_data, train_EA_MT_file)

    # ner_tagged_target_data = add_ner_tags(preprocessed_data, ner_target="target")
    # instructed_ner_tagged_target_data = add_task_instructions(ner_tagged_target_data, task_type="NER")
    # save_json(instructed_ner_tagged_target_data, train_NER_file)

 
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
    
    output_file = os.path.join(DATASET_PATH, "train.jsonl")
    save_json(all_data, output_file)
    print(f"Grouped all JSONL data into {output_file}")
