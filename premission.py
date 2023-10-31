from transformers import  AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import HfApi
import os
token = os.environ.get('hf_token')

#model_name = "Llama-2-7b-chat-hf"#"https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
model_name = "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8"

# hf_api = HfApi()
# user = hf_api.whoami(token)
# print(user)

tokenizer = AutoTokenizer.from_pretrained(model_name)#, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name)#, token=token)

