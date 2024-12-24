"""
Based on Entity-aware Multi-task Training Helps Rare Word Machine Translation paper, we use 6 filter techniques:
- unique parallel sentence filter
- equal source-target filter 
- multiple sources - one target and multiple targets - one source filters
- non-alphabetical filters
- repeating token filter;
- correct language filter
"""
import unicodedata
from langdetect import detect
from sacremoses import MosesPunctNormalizer

def unique_parallel_sentence_filter(data):
    unique_pairs = set()
    filtered_data = []
    for item in data:
        pair = (item["source"], item["target"])
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            filtered_data.append(item)

    return filtered_data

def equal_source_target_filter(data):
    filtered_data = []
    for item in data:
        if item["source"].strip() != item["target"].strip():
            filtered_data.append(item)

    return filtered_data

def multiple_sources_targets_filter(data):
    source_to_target = {}
    target_to_source = {}
    filtered_data = []

    for item in data:
        source, target = item['source'], item['target']
        if source in source_to_target and source_to_target[source] != target:
            continue
        if target in target_to_source and target_to_source[target] != source:
            continue

        source_to_target[source] = target
        target_to_source[target] = source
        filtered_data.append(item)

    return filtered_data

# def non_alphabetical_filter(data):
#     # Check if a character is alphabetic using Unicode categories (For japanese, arabian)
#     def is_alphabetic(char):
#         return unicodedata.category(char).startswith("L")
#     # Return True if sentence contains <= 50% non-alphabetical symbols
#     def is_valid_sentence(text):
#         total_chars = len(text)
#         alphabetic_chars = sum(1 for char in text if is_alphabetic(char))

#         return alphabetic_chars / max(total_chars, 1) >= 0.5
    
#     filtered_data = []
#     for item in data:
#         source_valid = is_valid_sentence(item['source'])
#         target_valid = is_valid_sentence(item['target'])

#         source_non_alpha = len([c for c in item['source'] if not is_alphabetic(c)])
#         target_non_alpha = len([c for c in item['target'] if not is_alphabetic(c)])
#         imbalance = max(source_non_alpha, target_non_alpha) / max(1, min(source_non_alpha, target_non_alpha, 1))

#         if source_valid and target_valid and imbalance <= 3:  # Allowable imbalance ratio
#             filtered_data.append(item)
#     return filtered_data

def non_alphabetical_filter(data):
    def is_alphabetic(char):
        category = unicodedata.category(char)
        return category.startswith("L") or category in {"Lo", "Lm", "Lt"}

    def non_alpha_ratio(text):
        total_chars = len(text)
        if total_chars == 0:
            return 1  # Treat empty text as 100% non-alphabetic
        non_alpha_count = sum(1 for char in text if not is_alphabetic(char))
        return non_alpha_count / total_chars

    filtered_data = []
    for item in data:
        source_text = item['source']
        target_text = item['target']

        # Calculate non-alphabetic symbol ratios
        source_ratio = non_alpha_ratio(source_text)
        target_ratio = non_alpha_ratio(target_text)

        # Calculate imbalance ratio (problem with Japanese)
        # source_non_alpha = sum(1 for char in source_text if not is_alphabetic(char))
        # target_non_alpha = sum(1 for char in target_text if not is_alphabetic(char))
        # imbalance_ratio = max(source_non_alpha, target_non_alpha) / max(1, min(source_non_alpha, target_non_alpha))

        # Apply filtering conditions
        # if source_ratio <= 0.5 and target_ratio <= 0.5 and imbalance_ratio <= 5:
        #     filtered_data.append(item)
        if source_ratio <= 0.5 and target_ratio <= 0.5:
            filtered_data.append(item)

    return filtered_data

def repeating_tokens_filter(data):
    def has_excessive_repetition(text):
        tokens = text.split()
        return len(set(tokens)) < len(tokens) / 2

    return [
        item
        for item in data
        if not has_excessive_repetition(item['source']) and not has_excessive_repetition(item['target'])
    ]

def correct_language_filter(data):
    filtered_data = []
    for item in data:
        source_lang = detect(item['source'])
        target_lang = detect(item['target'])
        if source_lang == item['source_locale'] and target_lang == item['target_locale']:
            filtered_data.append(item)
    return filtered_data

def punctuation_normalizer(data):
    normalizer = MosesPunctNormalizer()
    processed_data = []

    for item in data:
        item['source'] = normalizer.normalize(item['source'])
        item['target'] = normalizer.normalize(item['target'])
        processed_data.append(item)

    return processed_data

def apply_all_filterings(data):
    data = unique_parallel_sentence_filter(data)
    data = equal_source_target_filter(data)
    data = multiple_sources_targets_filter(data)
    data = non_alphabetical_filter(data)
    data = repeating_tokens_filter(data)
    data = correct_language_filter(data)
    data = punctuation_normalizer(data)
    return data 




