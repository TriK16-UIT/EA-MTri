from config import LOCALE_MAP

def add_task_instructions(data, task_type="translation"):
    preprocessed_data = []

    for item in data:
        source_locale = LOCALE_MAP.get(item['source_locale'], item['source_locale'])
        target_locale = LOCALE_MAP.get(item['target_locale'], item['target_locale'])

        if task_type == "NER":  # NER task
            task_instruction = f"recognize {source_locale} named entities: "
        elif task_type == "EA_MT":  # NE-aware MT task
            task_instruction = f"entity translate {source_locale} to {target_locale}: "
        else:  # Standard translation
            task_instruction = f"translate {source_locale} to {target_locale}: "

        tmp_item = item.copy()
        tmp_item['source'] = task_instruction + tmp_item['source']
        preprocessed_data.append(tmp_item)

    return preprocessed_data


