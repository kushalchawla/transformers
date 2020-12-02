import numpy as np
import json
import copy
import random

"""
This is Step 2 in the seq2seq pipeline.

Input: dtype.
Basically use the (context, scenario, utt) to create a series of templates (x) and utt(y). For test and valid data, we will only create the template from dgpt-m greedy approach.

"""

dtype = "train"
#template_types = ["copy", "greedy", "sample", "noisy"]
template_types = ["greedy"]

in_f = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/s2s_cxt_scen_utt.json"

in_f_greedy = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/dgpt_greedy.json"
with open(in_f_greedy, "r") as f:
	greedy_data = json.load(f)
print("loaded greedy data: ", in_f_greedy, len(greedy_data))

in_f_samples = ["/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/dgpt_sample1.json", "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/dgpt_sample2.json", "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/dgpt_sample3.json"]

sample_datas = []

for in_f_sample in in_f_samples:

    with open(in_f_sample, "r") as f:
        sample_datas.append(json.load(f))

print("loaded sample data ", in_f_samples)

out_src = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/" + dtype + "_" + "_".join(sorted(template_types)) + ".src"
out_tgt = "/project/glucas_540/kchawla/csci699/storage/data/seq2seq/casino/" + dtype + ".tgt"

def get_input(msg):

	msg = (" " + '<|endoftext|>' + " ").join(msg)

	msg = msg + " " + '<|endoftext|>' + " "
	return msg

def get_copy_templates(item):
    """
    autoencoding setup.
    """

    temps = []
    temps.append(item["utterance"])

    return temps

def get_greedy_templates(item):

    templates = []

    context = get_input(item['context'])

    templates.append(greedy_data[context])

    return templates

def get_sample_templates(item):
    
    templates = []

    context = get_input(item['context'])

    for ix in range(len(sample_datas)):
        templates.append(sample_datas[ix][context])

    return templates

def get_noisy_variants(temp):
    """
    basically switch item types and item numbers..
    """
    return []

def get_noisy_templates(templates):

    all_temps = []
    for temp in templates:
        noisy_variants = get_noisy_variants(temp)
        all_temps += noisy_variants

    return all_temps

def get_scenario_string(scenario):
    """
    rank corresponding to Food, Water, Firewood.

    permutation of 1 2 3
    """

    scenario_str = ""

    scenario_str += "High " + scenario["value2issue"]["High"] + " " + scenario["value2reason"]["High"] + " <|endoftext|> "

    scenario_str += "Medium " + scenario["value2issue"]["Medium"] + " " + scenario["value2reason"]["Medium"] + " <|endoftext|> "

    scenario_str += "Low " + scenario["value2issue"]["Low"] + " " + scenario["value2reason"]["Low"] + " <|endoftext|>"

    return scenario_str

def merge_scen_templates(scenario_str, templates):
    
    scen_temps = []

    for temp in templates:
        merged = scenario_str + " " + temp
        scen_temps.append(merged)

    return scen_temps

def get_pairs(item):
    """
    context, scenario, utterance
    """

    templates = []
    if("copy" in template_types):
        templates += get_copy_templates(item)

    if("greedy" in template_types):
        templates += get_greedy_templates(item)

    if("sample" in template_types):
        templates += get_sample_templates(item)

    if("noisy" in template_types):
        templates += get_noisy_templates(templates)

    templates = sorted(list(set(templates)))

    scenario_str = get_scenario_string(item['scenario'])

    utterance = item['utterance']

    scen_temps = merge_scen_templates(scenario_str, templates)

    pairs = []
    
    for merged in scen_temps:
        pair = [merged, utterance]
        pairs.append(pair)
    
    return pairs

with open(in_f) as f:
    all_data = json.load(f)
    all_data = all_data[dtype]

print("input: ", dtype, len(all_data))

#list of lists, each containing [src, tgt] pairs
out_data = []

for item in all_data:

    pairs = get_pairs(item)
    out_data += pairs

print("out_data: ", len(out_data))
print("sample: ", out_data[0])

with open(out_src, "w") as fsrc:
    with open(out_tgt, "w") as ftgt:

        for pair in out_data:
            fsrc.write(pair[0] + "\n")
            ftgt.write(pair[1] + "\n")

print("Output completed to: ", out_src, out_tgt)