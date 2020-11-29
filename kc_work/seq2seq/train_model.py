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

batch_size = modelparams["batch_size"]
train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                                drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Loading models ..")

if(globalparams["pretrained"]):
    #load pretrained encoder and pretrained decoder.
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(globalparams['pretrained_path'], globalparams['pretrained_path'])   
else:
    pass

model.to(device)

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=modelparams['lr'])
criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

num_train_batches = len(train_dataloader)
num_valid_batches = len(valid_dataloader)

print("num batches: ", num_train_batches, num_valid_batches)

def compute_loss(predictions, targets):
    """Compute our custom loss"""
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss

def train_model():
    model.train()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        labels = de_output.clone()
        out = model(input_ids=en_input, attention_mask=en_masks,
                                        decoder_input_ids=de_output, decoder_attention_mask=de_masks, labels=labels)
        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, de_output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print("Mean epoch loss:", (epoch_loss / num_train_batches))

def eval_model():
    model.eval()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        labels = de_output.clone()

        out = model(input_ids=en_input, attention_mask=en_masks,
                                        decoder_input_ids=de_output, decoder_attention_mask=de_masks, labels=labels)

        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, de_output)
        epoch_loss += loss.item()

    print("Mean validation loss:", (epoch_loss / num_valid_batches))

# MAIN TRAINING LOOP
for epoch in range(modelparams['num_epochs']):
    print("Starting epoch", epoch+1)
    train_model()
    eval_model()

print("Saving model ..")
save_location = modelparams['model_path']
if not os.path.exists(save_location):
    os.makedirs(save_location)

model.save_pretrained(save_location)