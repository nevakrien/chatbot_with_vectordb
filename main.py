from transformers import  AutoTokenizer,AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from emb import VectorDB
from collections import deque

import argparse
from os.path import join
# import os
# token = os.environ.get('hf_token')

#model_name = "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8"
#model_name="microsoft/DialoGPT-medium"#"HuggingFaceH4/zephyr-7b-beta"#"gpt2"
identety_prompt = "given the information provided and the context respond to the user"
CONTEXT=3

def get_next_text(model,tokenizer,input_messages):
	inputs=tokenizer.apply_chat_template(input_messages,return_tensors='pt',add_generation_prompt=True)
	#print(inputs)
	print(3*"\n"+10*'='+f'num tokens{inputs.shape}')
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

def cut_string(s, max_length):

    if len(s) <= max_length:
        return s

    cut_off_point = s.rfind(' ', 0, max_length)

    if cut_off_point == -1:
        return s[:max_length] + '...'

    return s[:cut_off_point] + '...'

# Test the function
test_string = "This is a test string to see if the function works correctly."
cut_string(test_string, 25)


def make_input(wiki_db: VectorDB,history: list ):
	prompt=history[-1]
	wiki=cut_string(wiki_db.search(prompt,5,add=False)[0],1000)
	wiki="\n\n".join(wiki)
	#print(wiki)
	messages=(
		[openai_format(identety_prompt+f'\nrelevent wikipedia information:\n {wiki}')]
		+[openai_format(x,'user' if i%2==(CONTEXT-1)%2 else 'assistant') for i,x in enumerate(history)]
		#+[openai_format(identety_prompt)]
		)
	print(messages)
	return messages

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="small chat app")
	parser.add_argument('--model_dir',  type=str, default=join("quantized_model","INT_8"),
	                    help='the directory of the saved model')

	parser.add_argument('--tokenizer_dir',  type=str, default=join("quantized_model","tokenizer"),
	                    help='the directory of the saved tokenizer')

	args = parser.parse_args()

	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
	model = OVModelForCausalLM.from_pretrained(args.model_dir)#,export=True)
	
	wiki_db=VectorDB.load('wiki_db.pkl')
	history=deque(maxlen=CONTEXT)

	next_output=['input text']
	while(True):
		prompt=input("\n"+next_output[0]+'\n'+10*"?")
		print(2*("\n"+10*"!"))
		
		history.append(prompt)
		input_messages = make_input(wiki_db,history)
		#print(input_messages)
		#print("\n")

		next_output=get_next_text(model,tokenizer,input_messages)
		print(2*("\n"+10*"!"))
