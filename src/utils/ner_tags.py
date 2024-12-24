from config import NER_TAGS, CLOSED_NER_TAGS, NER_TAGS_REPLACING_STYLE_1, NER_TAGS_REPLACING_STYLE_2, LOCALE_MAP_1, LOCALE_MAP_2
from src.utils.utils import remove_gap
import spacy

nlp_models = {}

def initialize_spacy_models():
    global nlp_models
    try:
        nlp_models['en'] = spacy.load("en_core_web_sm")
        target_languages = ['es', 'de', 'fr', 'it', 'ja'] 
        for lang in target_languages:
            nlp_models[lang] = spacy.load(f"{lang}_core_news_sm") 
    except OSError as e:
        print("Ensure the SpaCy models for the respective languages are installed.")
        raise e
    
def build_ner_dataset(data):
    preprocessed_data = []
    for item in data:
        tmp_item = item.copy()
        tmp_item["target"] = tmp_item["source"]
        tmp_item["target_locale"] = tmp_item["source_locale"]
        preprocessed_data.append(tmp_item)
    return preprocessed_data
    
def tag_named_entities(text, nlp):
    doc = nlp(text)
    tagged_text = text
    for ent in doc.ents:
        tagged_text = tagged_text.replace(ent.text, f"<{ent.label_}> {ent.text} </{ent.label_}>")
    return tagged_text

def remove_and_replace_tags(data, ner_target="target"):
    preprocessed_data = []

    def style_1(text):
        for old, new in NER_TAGS_REPLACING_STYLE_1:
            text = text.replace(old, new)
        text = remove_gap(text)
        return text
    
    def style_2(text):
        for old, new in NER_TAGS_REPLACING_STYLE_2:
            text = text.replace(old, new)
        text = remove_gap(text)
        return text

    for item in data:
        tmp_item = item.copy()
        text = item[ner_target]
        locale = item[f"{ner_target}_locale"]

        if locale in LOCALE_MAP_1:
            tmp_item[ner_target] = style_1(text)
        elif locale in LOCALE_MAP_2:
            tmp_item[ner_target] = style_2(text)

        preprocessed_data.append(tmp_item)
    
    return preprocessed_data

def remove_non_matching_tags(data):
    preprocessed_data = []

    for item in data:
        tmp_item = item.copy()
        source_sentence = tmp_item["source"]
        target_sentence = tmp_item["target"]

        source_tag_counts = {tag: source_sentence.count(tag) for tag in NER_TAGS}
        target_tag_counts = {tag: target_sentence.count(tag) for tag in NER_TAGS}

        if source_tag_counts != target_tag_counts:
            # Remove mismatched tags from sentences
            for tag, close_tag in zip(NER_TAGS, CLOSED_NER_TAGS):
                if source_tag_counts[tag] != target_tag_counts[tag]:
                    source_sentence = source_sentence.replace(tag, "").replace(close_tag, "")
                    target_sentence = target_sentence.replace(tag, "").replace(close_tag, "")

        source_sentence = remove_gap(source_sentence)
        target_sentence = remove_gap(target_sentence)
        tmp_item["source"] = source_sentence
        tmp_item["target"] = target_sentence
        preprocessed_data.append(tmp_item)

    return preprocessed_data

def add_ner_tags(data, ner_target="target"):
    preprocessed_data = []

    for item in data:
        text = item[ner_target]
        locale = item[f"{ner_target}_locale"]

        if locale not in nlp_models:
            print(f"Skipping {locale}: No SpaCy model found.")
            continue

        nlp = nlp_models[locale]
        tmp_item = item.copy()
        tmp_item[ner_target] = tag_named_entities(text, nlp)
        preprocessed_data.append(tmp_item)

    return preprocessed_data