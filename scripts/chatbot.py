from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_dir = "../gpt2-finetuned/checkpoint-18"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Bot is ready! Type your product question (type 'exit' to quit).")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input, max_new_tokens=64)
    print("Bot:", response[0]["generated_text"])
