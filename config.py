#PATH
RAW_TRAIN_PATH = "Data/raw/train"
PREPROCESSED_TRAIN_PATH = "Data/preprocessed/train"
DATASET_PATH = "Dataset"
#CONST
LOCALE_MAP = {
    'en': 'English',
    'ja': 'Japanese',
    'es': 'Spanish',
    'de': 'German',
    'fr': 'French',
    'it': 'Italian',
    'ar': 'Arabic'
}
NER_TAGS = [
        "<CARDINAL>", "<DATE>", "<EVENT>", "<FAC>", "<GPE>", "<LANGUAGE>", "<LAW>", "<LOC>",
        "<MONEY>", "<NORP>", "<ORDINAL>", "<ORG>", "<PERCENT>", "<PERSON>", "<PRODUCT>", 
        "<QUANTITY>", "<WORK_OF_ART>", "<TIME>"
    ]
CLOSED_NER_TAGS = [
        "</CARDINAL>", "</DATE>", "</EVENT>", "</FAC>", "</GPE>", "</LANGUAGE>", "</LAW>", "</LOC>",
        "</MONEY>", "</NORP>", "</ORDINAL>", "</ORG>", "</PERCENT>", "</PERSON>", "</PRODUCT>", 
        "</QUANTITY>", "</WORK_OF_ART>", "</TIME>"
    ]
LOCALE_MAP_1 = ["en", "ja"]
LOCALE_MAP_2 = ["es", "de", "fr", "it", "ar"]
NER_TAGS_REPLACING_STYLE_1 = [
            ("<GPE>", "<LOC>"), ("</GPE>", "</LOC>"),
            ("<CARDINAL>", ""), ("</CARDINAL>", ""),
            ("<DATE>", ""), ("</DATE>", ""),
            ("<EVENT>", ""), ("</EVENT>", ""),
            ("<FAC>", ""), ("</FAC>", ""),
            ("<LANGUAGE>", ""), ("</LANGUAGE>", ""),
            ("<LAW>", ""), ("</LAW>", ""),
            ("<MONEY>", ""), ("</MONEY>", ""),
            ("<NORP>", ""), ("</NORP>", ""),
            ("<ORDINAL>", ""), ("</ORDINAL>", ""),
            ("<PERCENT>", ""), ("</PERCENT>", ""),
            ("<PRODUCT>", ""), ("</PRODUCT>", ""),
            ("<QUANTITY>", ""), ("</QUANTITY>", ""),
            ("<WORK_OF_ART>", ""), ("</WORK_OF_ART>", ""),
            ("<TIME>", ""), ("</TIME>", ""),
            ("<PET_NAME>", ""), ("</PET_NAME>", ""),
            ("<PHONE>", ""), ("</PHONE>", ""),
            ("<TITLE_AFFIX>", ""), ("</TITLE_AFFIX>", "")
        ]

NER_TAGS_REPLACING_STYLE_2 = [
            ("<PER>", "<PERSON>"), ("</PER>", "</PERSON>"),
            ("<MISC>", ""), ("</MISC>", "")
        ]