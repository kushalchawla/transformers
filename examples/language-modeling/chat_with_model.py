from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
from transformers import GPT2LMHeadModel
import torch 

logdir = "../../../../storage/logs/best-model"

tokenizer = AutoTokenizer.from_pretrained(logdir)
model = AutoModelWithLMHead.from_pretrained(logdir).cuda()

def get_input(msg):
	msg = msg.strip("\n")
	return msg

def format_response(msg):
	msg = msg + "\n"
	return msg

def generate_response(user_input):
	user_input = get_input(user_input)
	user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
	response_ids = model.generate(user_input_ids.cuda(), max_length=1024, do_sample=True, top_k=20, top_p=0.95, pad_token_id=tokenizer.pad_token_id, use_cache=False)
	response = tokenizer.decode(response_ids[:, user_input_ids.shape[-1]:][0], skip_special_tokens=True)                                                                                          
	output = format_response(response)
	return output

print("Model Loaded.")

while(True):
	user_input = input()
	print(generate_response(user_input))
 
