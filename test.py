from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small", use_auth_token=False, trust_remote_code=False)
print(model.config)

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
