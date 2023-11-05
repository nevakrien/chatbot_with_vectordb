from optimum.intel import OVQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from os.path import join

import argparse

def quantize(model_name,save_dir):
	pt_model = AutoModelForCausalLM.from_pretrained(model_name)
	quantizer = OVQuantizer.from_pretrained(pt_model)
	quantizer.quantize(save_directory=join(save_dir,"INT_8"), weights_only=True)

	tokenizer=AutoTokenizer.from_pretrained(model_name)
	tokenizer.save_pretrained(join(save_dir,'tokenizer'))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description="takes a model from huggingface or local cach and quntizes it")
	parser.add_argument('--model_name',  type=str, default="meta-llama/Llama-2-7b-hf",
	                    help='the name of the model to be quntized')

	parser.add_argument('--save_dir' ,type=str, default='quantized_model',
	                    help='saving directory where the weights be stored')

	args = vars(parser.parse_args())

	quantize(**args)
	print('done')
