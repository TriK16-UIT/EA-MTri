import os
import torch
import evaluate
import numpy as np
from utils.utils import load_json
from torch.utils.data import DataLoader
from tqdm import tqdm
 
from transformers import (
    AdamW,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from datasets import load_dataset
from datasets import Dataset

MODEL = 'google/mt5-small'
BATCH_SIZE = 4
NUM_PROCS = 16
EPOCHS = 10
OUT_DIR = 'results_mt5small_mt'
MAX_LENGTH = 256 

bleu = evaluate.load("bleu")

model = MT5ForConditionalGeneration.from_pretrained(MODEL)
tokenizer = T5Tokenizer.from_pretrained(MODEL, use_fast=True)

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
    with open(OUT_DIR+"/devel-predictions.txt", "w") as f:
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

data = load_json("Data/preprocessed/train/de/train_NER.jsonl")

#test
# data = data[0:100]

file_data = {"src": [item["source"] for item in data], "trg": [item["target"] for item in data]}
dataset = Dataset.from_dict(file_data)

train_test_split = dataset.train_test_split(test_size=0.02, seed=42)
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

# # train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

# # Define optimizer
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# def train(model, train_dataloader, optimizer, epoch):
#     model.train()
#     total_loss = 0
    
#     for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
#         optimizer.zero_grad()
        
#         # Move batch to DEVICE
#         input_ids = batch['input_ids'].to(DEVICE)
#         attention_mask = batch['attention_mask'].to(DEVICE)
#         labels = batch['labels'].to(DEVICE)

#         # Forward pass
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#     avg_loss = total_loss / len(train_dataloader)
#     print(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
#     return avg_loss

# # Validation loop
# def evaluate(model, val_dataloader):
#     model.eval()
#     total_loss = 0
#     predictions, references = [], []

#     with torch.no_grad():
#         for batch in tqdm(val_dataloader, desc="Evaluating"):
#             # Move batch to DEVICE
#             input_ids = batch['input_ids'].to(DEVICE)
#             attention_mask = batch['attention_mask'].to(DEVICE)
#             labels = batch['labels'].to(DEVICE)

#             # Forward pass
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()

#             # Generate predictions
#             generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH)

#             predictions.extend(generated_tokens.cpu().numpy())
#             references.extend(labels.cpu().numpy())

#     predictions = np.array(predictions)
#     references = np.array(references)

#     # Compute metrics
#     # predictions = np.concatenate(predictions)
#     # references = np.concatenate(references)
#     metrics = compute_metrics((predictions, references))

#     avg_loss = total_loss / len(val_dataloader)
#     print(f"Validation Loss: {avg_loss:.4f}, BLEU: {metrics['bleu']:.4f}")

#     return avg_loss, metrics

# # Training and evaluation
# num_epochs = EPOCHS
# for epoch in range(1, num_epochs + 1):
#     train_loss = train(model, train_dataloader, optimizer, epoch)
#     val_loss, val_bleu = evaluate(model, val_dataloader)

#     # Save model checkpoint
#     torch.save(model.state_dict(), f"{OUT_DIR}/mt5_epoch_{epoch}.pt")

training_args = Seq2SeqTrainingArguments(
    run_name=MODEL,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.0001,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=EPOCHS,
    logging_dir=OUT_DIR,
    output_dir=OUT_DIR,
    generation_max_length=128,
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    predict_with_generate=True,
    metric_for_best_model='bleu'    
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