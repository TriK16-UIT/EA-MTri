from src.utils.ner_tags import build_ner_dataset

input_data = [
    {"id": "a9011ddf", "source": "What is the <GPE> seventh tallest mountain </GPE> in North America?", "target": "Wie heißt der siebthöchste Berg <LOC> Nordamerikas?</LOC>", "source_locale": "en", "target_locale": "de"},
    {"id": "bff78c91", "source": "What year was the first <DATE> book </DATE> of the A Song of Ice and Fire series published?", "target": "In welchem Jahr wurde das erste Buch der Reihe \"Das Lied von Eis und Feuer\" veröffentlicht?", "source_locale": "en", "target_locale": "de"}
]

output = build_ner_dataset(input_data)
print(output)