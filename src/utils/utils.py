import json

def load_json(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def remove_gap(text):
    text = " ".join(text.split())
    return text


