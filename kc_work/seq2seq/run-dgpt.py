import numpy as np
import json
import copy
import random
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
import torch

dgpt_dir = "/project/glucas_540/kchawla/csci699/storage/logs/best-model"
tokenizer = AutoTokenizer.from_pretrained(dgpt_dir)
model = AutoModelWithLMHead.from_pretrained(dgpt_dir).cuda()

print("tokenizer and model loaded from: ", dgpt_dir)

def get_input(msg):

	msg = (" " + tokenizer.eos_token + " ").join(msg)

	msg = msg + " " + tokenizer.eos_token + " "
	return msg

def format_response(msg):
	msg = msg.replace("<|endoftext|>", "").strip()
	return msg

def generate_response(context):
	context = get_input(context)
	context_ids = tokenizer.encode(context + tokenizer.eos_token, return_tensors='pt')

	if(strategy == "greedy"):
		response_ids = model.generate(context_ids.cuda(), max_length=1024, pad_token_id=tokenizer.pad_token_id)
	else:
		response_ids = model.generate(context_ids.cuda(), max_length=1024, do_sample=True, top_k=20, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
	
	response = tokenizer.decode(response_ids[:, context_ids.shape[-1]:][0])                                                                                          
	output = format_response(response)
	return output

in_f = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/s2s_cxt_scen_utt.json"
with open(in_f) as f:
	all_data = json.load(f)

print("data loaded from: ", in_f)

strategy = "sample3" #greedy/sample
out_data = {}
out_f = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/dgpt_" + strategy + ".json"

for dtype in ["train", "valid", "test"]:
	print(dtype)

	total = len(all_data[dtype])
	for ix, item in enumerate(all_data[dtype]):
		if(ix%100 == 0):
			print(ix, total)
		response = generate_response(item['context'])
		out_data[get_input(item['context'])]  = response
	
with open(out_f, "w") as f:
	json.dump(out_data, f)

print("Output saved: ", out_f)