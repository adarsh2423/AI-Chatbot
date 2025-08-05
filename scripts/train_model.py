from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_from_disk("data/processed_dataset")

def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()   # This tells the Trainer to use next-token prediction for loss
    return tokens

tokenized_dataset = train_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="../gpt2-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=False  # Use True only if using compatible GPUs (NVIDIA/ROCm)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()
