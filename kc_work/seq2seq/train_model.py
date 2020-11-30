import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import TranslationDataset
from transformers import BertTokenizerFast, BertTokenizer
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
import math

# Identify the config file
if len(sys.argv) < 2:
    print("No config file specified. Using the default config.")
    configfile = "config.json"
else:
    configfile = sys.argv[1]

# Read the params
with open(configfile, "r") as f:
    config = json.load(f)

globalparams = config["global_params"]
modelparams = config["model_params"]

# Load the tokenizers
en_tokenizer = BertTokenizer.from_pretrained(globalparams["tokenizer_path"])
de_tokenizer = BertTokenizer.from_pretrained(globalparams["tokenizer_path"])

# Init the dataset
train_en_file = globalparams["train_en_file"]
train_de_file = globalparams["train_de_file"]
valid_en_file = globalparams["valid_en_file"]
valid_de_file = globalparams["valid_de_file"]
test_en_file = globalparams["test_en_file"]
test_de_file = globalparams["test_de_file"]

batch_size = modelparams["batch_size"]
train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)

test_dataset = TranslationDataset(test_en_file, test_de_file, en_tokenizer, de_tokenizer)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=test_dataset.collate_function)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def compute_loss(predictions, targets, criterion, perplexity=False):
    """Compute our custom loss"""
    #print("Compute loss: ")
    #print("inputs, preds, targets", predictions.shape, targets.shape)
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    #print("preds, targets", predictions.shape, targets.shape)

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    #print(rearranged_output.shape, rearranged_target.shape)
    #print(rearranged_target)

    loss = criterion(rearranged_output, rearranged_target)

    if(not perplexity):
        #means that criterion passed in mean reduction, and currently training is going on.
        return loss
    else:
        #eval mode is going on...criterion has sum reduction currently.
        return loss, (rearranged_target != 0).sum()

def train_model(model, optimizer, criterion):

    model.train()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        #print(en_input.shape, de_output.shape, en_masks.shape, de_masks.shape)

        labels = de_output.clone()
        out = model(input_ids=en_input, attention_mask=en_masks,
                                        decoder_input_ids=de_output, decoder_attention_mask=de_masks, labels=labels)
        
        #print(len(out))
        #print(out[0].shape)
        #print(out[1].shape)
        prediction_scores = out[1]
        #print("pred scores: ", prediction_scores.shape)
        predictions = F.log_softmax(prediction_scores, dim=2)
        #print("predictions: ", predictions.shape, predictions)
        #print("output: ", de_output.shape, de_output)
        loss = compute_loss(predictions, de_output, criterion)
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print("Mean train loss:", (epoch_loss / len(train_dataloader)))

def eval_model(model, criterion, datatype="valid", perplexity=False):

    dataloader = None
    if(datatype == "train"):
        dataloader = train_dataloader
    elif(datatype == "valid"):
        dataloader = valid_dataloader
    elif(datatype == "test"):
        dataloader = test_dataloader
    else:
        raise AssertionError

    model.eval()

    if(perplexity):
        loss_sum = 0.0
        count_eles = 0
    else:
        epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(dataloader):

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        labels = de_output.clone()

        with torch.no_grad():
            out = model(input_ids=en_input, attention_mask=en_masks,
                                            decoder_input_ids=de_output, decoder_attention_mask=de_masks, labels=labels)
            prediction_scores = out[1]
            predictions = F.log_softmax(prediction_scores, dim=2)

        if(perplexity):
            loss, eles = compute_loss(predictions, de_output, criterion, perplexity=perplexity)
            loss_sum += loss.item()
            count_eles += eles.item()
        else:
            loss = compute_loss(predictions, de_output, criterion, perplexity=perplexity)
            epoch_loss += loss.item()
    
    if(perplexity):
        #eval mode.
        mean_nll = loss_sum / count_eles
        ppl = math.exp(mean_nll)
        print("Perplexity: ", datatype, ppl)
    else:
        #training going on
        print("Mean loss", datatype, (epoch_loss / len(dataloader)))

if(globalparams["do_train"]):
    #load model from pretrained/scratch and train it/save it in the provided dir.
    print("TRAIN MODE: ")

    if(globalparams["pretrained"]):
        #load pretrained encoder and pretrained decoder.
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(globalparams['pretrained_path'], globalparams['pretrained_path'])
        print("pretrained model loaded.", globalparams["pretrained_path"])
    else:
        pass

    model.to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=modelparams['lr'])
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    num_train_batches = len(train_dataloader)
    num_valid_batches = len(valid_dataloader)

    print("num batches: ", num_train_batches, num_valid_batches)

    # MAIN TRAINING LOOP
    for epoch in range(modelparams['num_epochs']):
        print("Starting epoch", epoch+1)
        train_model(model, optimizer, criterion)
        eval_model(model, criterion)

    print("Saving model ..")
    save_location = modelparams['model_path']
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    model.save_pretrained(save_location)

if(globalparams["do_eval"]):
    #load the trained encoder decoder model from the provided dir and then evaluate it on all three datasets on perplexity.

    print("EVAL MODE: ")
    model = EncoderDecoderModel.from_pretrained(modelparams['model_path'])
    print("Trained EncDec Model loaded: ", modelparams["model_path"])
    model.to(device)

    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id, reduction='sum')

    #TODO evaluate perplexity on each.
    eval_model(model, criterion, datatype="train", perplexity=True)
    eval_model(model, criterion, datatype="valid", perplexity=True)
    eval_model(model, criterion, datatype="test", perplexity=True)