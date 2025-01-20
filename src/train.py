import torch
import evaluate
import numpy as np
from utils.utils import load_json
 
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from datasets import Dataset

from config import DATASET_PATH


#SETTINGS
MODEL = 'google/mt5-small'
BATCH_SIZE = 4
NUM_PROCS = 16
EPOCHS = 5
OUT_DIR = 'results_mt5small_mt'
MAX_LENGTH = 256 

bleu = evaluate.load("bleu")

model = MT5ForConditionalGeneration.from_pretrained(MODEL)
tokenizer = T5Tokenizer.from_pretrained(MODEL)

# new tokens
new_tokens = ["<LOC>","</LOC>","<ORG>","</ORG>","<PERSON>","</PERSON>"]
# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))
# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(f"\nCalculating BLEU\n")
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    with open(OUT_DIR+"/devel-predictions.txt", "w", encoding="utf-8") as f:
        for sentence in pred_str:
            f.write(sentence + "\n")
    return {"bleu": round(np.mean(bleu_output["bleu"])*100, 2)}

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["src"], text_target=batch["trg"], padding='max_length', truncation=True, max_length=MAX_LENGTH)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = inputs.labels
  return batch


data = load_json(DATASET_PATH + "train.jsonl")

#test
# data = data[0:100]

file_data = {
    "src": [item["src"] for item in data],
    "trg": [item["trg"] for item in data]
}
dataset = Dataset.from_dict(file_data)

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_data = train_test_split["train"]
val_data = train_test_split["test"]

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns=["src", "trg"],
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns=["src", "trg"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

print(val_data)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)
model.to(DEVICE)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    logging_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=0.0001,
    weight_decay=0.01,
    generation_max_length=128,
    load_best_model_at_end=True,
    predict_with_generate=True,
    auto_find_batch_size=True,
    metric_for_best_model='bleu',
    run_name=MODEL,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

history = trainer.train()