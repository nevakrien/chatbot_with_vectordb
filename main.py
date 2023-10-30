from transformers import  AutoTokenizer,AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from emb import VectorDB

# import os
# token = os.environ.get('hf_token')

model_name = "HuggingFaceH4/zephyr-7b-beta"#"gpt2"
identety_prompt = "be cheary and help people try and talk about python"

def get_next_text(model,tokenizer,input_messages):
	inputs=tokenizer.apply_chat_template(input_messages,return_tensors='pt',add_generation_prompt=True)
	#print(inputs)
	output=model.generate(inputs,do_sample=True,num_beams=3,top_k=10,no_repeat_ngram_size=2,
		max_new_tokens=100,temperature=2.,penalty_alpha=0.6,early_stopping=True)
	#print(output)
	output=tokenizer.batch_decode(output[:,inputs.shape[-1]:],skip_special_tokens=True)
	#output=[x.split("<|assistant|>")[-1] for x in output]
	return output

#this is the standard format introduced by openai and used by hf
def openai_format(text,role='system'):
    return {'role':role,'content':text}
#assistant
#user

def make_input(user_db,bot_db,prompt):
	messages=(
		[openai_format(identety_prompt),openai_format('user said in the past:')]
		+[openai_format(x,'user') for x in user_db.search([prompt],3,add=True)[0]]
		+[openai_format('I the bot said in the past')]
		+[openai_format(x,'assistant') for x in bot_db.search([prompt],3,add=False)[0]]
		+[openai_format('user is saying:'),openai_format(prompt,'user')]
		)

	return messages

if __name__=='__main__':
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)#,export=True)
	
	user_db=VectorDB()
	bot_db=VectorDB()

	next_output=['input text']
	while(True):
		prompt=input(next_output[0]+'\n')
		print(2*("\n"+10*"!"))
		
		input_messages = make_input(user_db,bot_db,prompt)
		#print(input_messages)
		#print("\n")

		next_output=get_next_text(model,tokenizer,input_messages)
		bot_db.add(next_output[0])

		print(2*("\n"+10*"!"))
