from transformers import  AutoTokenizer,AutoModelForCausalLM

model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [
{
    "role": "system",
    "content": "You are a friendly chatbot who always responds in the style of a pirate",
},
{"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors='pt')
print(tokens)
