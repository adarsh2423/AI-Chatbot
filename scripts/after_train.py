from transformers import AutoTokenizer

model_dir = "../gpt2-finetuned/checkpoint-18" # adapt to your model dir
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # original source tokenizer
tokenizer.save_pretrained(model_dir)
