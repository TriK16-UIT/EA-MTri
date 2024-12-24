import sys

# Hàm đếm và so khớp các thẻ NER giữa câu nguồn và câu đích
def process_ner_tags(source_file, target_file):
    # Các thẻ NER
    tags = [
        "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
        "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY",
        "WORK_OF_ART", "TIME"
    ]

    # Mở tệp nguồn và đích
    with open(source_file, 'r', encoding='utf-8') as src, \
         open(target_file, 'r', encoding='utf-8') as trg, \
         open(source_file + ".match", 'w', encoding='utf-8') as out_src_match, \
         open(target_file + ".match", 'w', encoding='utf-8') as out_trg_match, \
         open(source_file + ".nonmatch", 'w', encoding='utf-8') as out_src_nonmatch, \
         open(target_file + ".nonmatch", 'w', encoding='utf-8') as out_trg_nonmatch:

        mismatch_count = 0
        removed_tag_count = 0

        # Xử lý từng cặp câu
        for source_sentence, target_sentence in zip(src, trg):
            source_sentence = source_sentence.strip()
            target_sentence = target_sentence.strip()

            # Đếm số lượng thẻ NER trong câu nguồn và câu đích
            source_counts = {tag: source_sentence.count(f"<{tag}>") for tag in tags}
            target_counts = {tag: target_sentence.count(f"<{tag}>") for tag in tags}

            # Tính tổng số thẻ trong cả hai câu
            total_source_tags = sum(source_counts.values())
            total_target_tags = sum(target_counts.values())

            # Nếu khớp, lưu câu vào tệp "match"
            if source_counts == target_counts:
                out_src_match.write(source_sentence + "\n")
                out_trg_match.write(target_sentence + "\n")
            else:
                # Ghi câu gốc vào tệp "nonmatch"
                out_src_nonmatch.write(source_sentence + "\n")
                out_trg_nonmatch.write(target_sentence + "\n")
                mismatch_count += 1

                # Loại bỏ các thẻ không khớp
                for tag in tags:
                    if source_counts[tag] != target_counts[tag]:
                        source_sentence = source_sentence.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
                        target_sentence = target_sentence.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
                        removed_tag_count += 1

                # Ghi câu đã loại bỏ thẻ vào tệp "match"
                out_src_match.write(source_sentence + "\n")
                out_trg_match.write(target_sentence + "\n")

        print(f"Found {mismatch_count} sentence pairs with a NER tag count mismatch between source and target sentences.")
        print(f"Removed {removed_tag_count} NER tags in sentence pairs with a NER tag count mismatch between source and target sentences.")

# Chạy script với file ví dụ
if __name__ == "__main__":
    # Ví dụ: Tên tệp câu nguồn và câu đích
    source_file = "source_sentences.txt"
    target_file = "target_sentences.txt"

    process_ner_tags(source_file, target_file)
