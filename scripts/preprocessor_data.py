import json
from datasets import Dataset

# Load data
with open("../data/products.json") as f:
    products = json.load(f)

examples = []
for item in products:
    q = f"Tell me about {item['product']}"
    a = f"Product: {item['product']}\nDescription: {item['description']}\nPrice: {item['price']}"
    examples.append({"text": f"{q} [SEP] {a}"})

# Save as Hugging Face Dataset (optional)
dataset = Dataset.from_list(examples)
dataset.save_to_disk("data/processed_dataset")
