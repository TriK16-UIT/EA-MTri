from src.utils.ner_tags import remove_non_matching_tags

input_data = [
    {"id": "a9011ddf", "source": "Today we are <ORG>hearing</ORG> the case of <PER> Albin Kurti </PER> of <LOC> Kosovo </LOC>.", "target": "Wir haben heute von dem Fall <PER> Albin Kurti </PER> aus dem <LOC> Kosovo </LOC> erfahren.", "entities": ["Q49"], "from": "mintaka"}
]

output = remove_non_matching_tags(input_data)
print(output)
print(input_data)