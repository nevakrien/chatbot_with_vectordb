from transformers import  AutoTokenizer,AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from emb import VectorDB
	
model_name = 'gpt2'#"HuggingFaceH4/zephyr-7b-beta"
starting_prompt = "be cheary and help people start with your name"

def get_next_text(model,tokenizer,input_text):
	inputs=tokenizer([input_text],return_tensors='pt')
	output=model.generate(**inputs,max_new_tokens=100)
	print(output)
	return tokenizer.batch_decode(output,skip_special_tokens=True)


def make_input(user_db,bot_db,prompt):
	return f"""
	user said: {[prompt]+user_db.search([prompt],3,add=True)} 

	I (the bot) said {bot_db.search([prompt],3,add=True)}
	"""

if __name__=='__main__':
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)#,export=True)
	
	user_db=VectorDB()
	bot_db=VectorDB()
	

	prompt=starting_prompt
	while(True):
		input_text = make_input(user_db,bot_db,prompt)
		print(input_text)

		next_output=get_next_text(model,tokenizer,input_text)[0]
		print(10*("\n"+10*"!"))
		prompt=input(next_output)
		print("\n")
