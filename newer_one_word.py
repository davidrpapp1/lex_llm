# Imported libraries
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
import tiktoken
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase

torch.manual_seed(1337)

# Needs mask for padding tokens in encoder self attention and mask for padding at encoder-decoder attention

# Enable/disable training mode and the reprocessing of training data located in reservoir
training_mode = True
reprocess_data = False
small_batch_validation_output = True # Displays snapshot of tensor indices during vectorisation process
train_val_split = 0.9 # fraction of data used for training

#encoding = tiktoken.get_encoding("cl100k_base")
#vocab_size=100256

# Miscellaneous parameters
MAX_LENGTH = 10 # Maximum sentence length to keep from training data
MIN_COUNT = 1 # Keep any words from training data that show up equal to or more than MIN_COUNT times


# Configure models
model_name = 'lex_llm' # For file saving label
eval_interval = 10
eval_iters = 3
n_embed = 512
n_head = 8
n_layer = 6
dropout = 0.1
batch_size = 8
block_size = 312 #1152
max_iters = 2000
learning_rate = 0.0001



# Use CUDA if installed on current system, otherwise use CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

src_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
src_tokenizer.pre_tokenizer = Whitespace()
src_tokenizer.normalizer = Lowercase()
trainer = WordLevelTrainer(special_tokens=["<unk>", "<sos>", "<eos>", "<pad>"], min_frequency=2)
src_tokenizer.train(["D:/robo_data/train.source"], trainer=trainer)
src_vocab_size = src_tokenizer.get_vocab_size()

tgt_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
tgt_tokenizer.pre_tokenizer = Whitespace()
tgt_tokenizer.normalizer = Lowercase()
trainer = WordLevelTrainer(special_tokens=["<unk>", "<sos>", "<eos>", "<pad>"], min_frequency=2)
tgt_tokenizer.train(["D:/robo_data/train.target"], trainer=trainer)
tgt_vocab_size = tgt_tokenizer.get_vocab_size()

pairs1 = []
pairs2 = []
pairs = []
with open("D:/robo_data/train.source", "r", encoding="UTF-8") as f:
    for line in f:
        if len(pairs1) < 500000:
            pairs1.append(line.lower())
with open("D:/robo_data/train.target", "r", encoding="UTF-8") as f:
    for line in f:
        if len(pairs2) < 500000:
            pairs2.append(line.lower())
for i in range(len(pairs1)):
    pairs.append([pairs1[i], pairs2[i]])


datain=[]
inmask=[]
dataout=[]
outmask=[]
for q in range(len(pairs)):
    datain_e = src_tokenizer.encode(pairs[q][0]).ids
    dataout_e = tgt_tokenizer.encode(pairs[q][1]).ids

    datain_e.insert(0, src_tokenizer.token_to_id("<sos>"))
    datain_e.append(src_tokenizer.token_to_id("<eos>"))
    if len(datain_e) < block_size:
        incm = [1]*len(datain_e) + [0]*(block_size - len(datain_e))
        datain_e.extend([src_tokenizer.token_to_id("<pad>")]*(block_size - len(datain_e)))
    else:
        incm = [1]*len(datain_e)

    dataout_e.insert(0, tgt_tokenizer.token_to_id("<sos>"))
    #rand = random.randint(1, len(dataout_e))
    #dataout_e = dataout_e[:rand]
    dataout_e.append(tgt_tokenizer.token_to_id("<eos>"))
    
    if len(dataout_e) < block_size:
        outcm = [1]*len(dataout_e) + [0]*(block_size - len(dataout_e))
        dataout_e.extend([tgt_tokenizer.token_to_id("<pad>")]*(block_size - len(dataout_e)))
    else:
        outcm = [1]*len(dataout_e)

    '''    
    if len(datain_e) > block_size:
        block_size = len(datain_e)
    if len(dataout_e) > block_size:
        block_size = len(dataout_e)
print(block_size)
'''

    datain.append(torch.tensor(datain_e, dtype=torch.long))
    inmask.append(torch.tensor(incm, dtype=torch.int))
    dataout.append(torch.tensor(dataout_e, dtype=torch.long))
    outmask.append(torch.tensor(outcm, dtype=torch.int))

print(pairs[5][0])
print(datain[5])
print(inmask[5])
print(pairs[5][1])
print(dataout[5])
print(outmask[5])
print(pairs[7][0])
print(datain[7])
print(inmask[7])
print(pairs[7][1])
print(dataout[7])
print(outmask[7])


n=int(train_val_split*len(datain))
datain_t = datain[:n]
inmask = inmask[:n]
datain_v = datain[n:]
dataout_t = dataout[:n]
outmask = outmask[:n]
dataout_v = dataout[n:]

def get_batch(split):
    if split == "train":
        data_i = datain_t     
        data_o = dataout_t
    else:
        data_i = datain_v
        data_o = dataout_v
    ix = torch.randint(len(data_i), (batch_size,))
    xin = torch.stack([data_i[i][:block_size-1] for i in ix])
    xout = torch.stack([data_o[i][:block_size-1] for i in ix])
    yin = torch.stack([data_i[i][1:block_size] for i in ix])
    yout = torch.stack([data_o[i][1:block_size] for i in ix])

    min = torch.stack([inmask[i][:block_size-1] for i in ix])
    mout = torch.stack([outmask[i][:block_size-1] for i in ix])
    xin, xout, yin, yout = xin.to(device), xout.to(device), yin.to(device), yout.to(device)
    return xin, xout, yin, yout, min, mout

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xin, Xout, Yin, Yout, minf, moutf = get_batch(split)
            proj_output = model(Xin, Xout, minf, moutf)
            losses[k] = loss_fn(proj_output.view(-1,tgt_tokenizer.get_vocab_size()), Yout.view(-1))
        out[split] = losses.mean()
    model.train()
    return out


class EncoderHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, min):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(min.unsqueeze(1)==0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #weighted aggregation of values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class EncoderMultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([EncoderHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, min):
        out = torch.cat([h(x, min) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out




class NoMaskHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, min):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(enc_out)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(min.unsqueeze(1)==0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #weighted aggregation of values
        v = self.value(enc_out)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class NoMaskMultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([NoMaskHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, min):
        out = torch.cat([h(x, enc_out, min) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mout):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
        wei = wei.masked_fill(mout.unsqueeze(1) == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #weighted aggregation of values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mout):
        out = torch.cat([h(x, mout) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    
    def __init__(self,n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = EncoderMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, min):
        att = self.sa(x, min)
        x = self.ln1(att + x)
        ff = self.ffwd(x)
        out = self.ln2(ff + x)
        return out, min
    
class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ca = NoMaskMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.lnc = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, enc_out, min, mout):

        att = self.sa(x, mout)
        x = self.ln1(att + x)
        catt = self.ca(x, enc_out, min)
        x = self.lnc(catt + x)
        ff = self.ffwd(x)
        out = self.ln2(ff + x)
        return out, enc_out, min, mout
    

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
class LangMod(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(tgt_vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.src_token_embedding_table = nn.Embedding(src_vocab_size, n_embed)
        #self.src_position_embedding_table = nn.Embedding(block_size, n_embed)
        self.encoderblocks = mySequential(*[EncoderBlock(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.blocks = mySequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #can be RMS norm
        self.lm_head = nn.Linear(n_embed, tgt_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, enc_out, idx, min, mout):
        B,T = idx.shape
        Be, Te = enc_out.shape

        tok_emb_inp = self.src_token_embedding_table(enc_out)
        pos_emb_inp = self.position_embedding_table(torch.arange(Te, device=device))
        x_input = tok_emb_inp + pos_emb_inp

        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb # (B,T,C)

        x_enc, min = self.encoderblocks(x_input, min)
        x, x_enc, min, mout = self.blocks(x, x_enc, min, mout) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        proj_output = self.lm_head(x) #(B,T,vocab_size)

        return proj_output
    
    def generate(self, inp, idx, inmask, max_new_tokens):
        for i in range(1, max_new_tokens):
            #outmask = torch.tensor([[1]*i + [0]*(max_new_tokens-i)], dtype=torch.int)
            #idx_cond = torch.cat((idx[:, :i], outmask[:, i:]), dim=1)

            idx_cond = idx
            outmask = torch.tensor([[1]*i])

            #print(outmask.shape, idx.shape, idx_cond.shape)
            proj_output = self(inp, idx_cond, inmask, outmask)

            logits = proj_output[:, -1, :] # last time step (B,C)
            probs = F.log_softmax(logits, dim=-1) # (B,C)
            idx_next = torch.max(probs, dim=1).indices.unsqueeze(0) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if idx_next[0] == tgt_tokenizer.token_to_id("<eos>"):
                break

        return idx
    
    
model = LangMod()
loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("<pad>"), label_smoothing=0.1).to(device)


print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    model.train()

    xbi, xbo, ybi, ybo, min, mout = get_batch("train")
    #print(xbi)
    #print(xbo)
    #print(ybo)

    proj_output = model(xbi, xbo, min, mout)
    loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), ybo.view(-1))

    if iter % eval_interval == 0:
        torch.save({
            "iter": iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, "D:/checkpoint.pt")

    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
on = True
while on:
    q = input()
    if q == "quit":
        on = False
    else:
        gen = torch.tensor([[tgt_tokenizer.token_to_id("<sos>")]], dtype=torch.long, device=device)
        encodedin = src_tokenizer.encode("<sos> "+q.lower()).ids
        #if len(encodedin) < block_size-1:
        #    inputmask = torch.tensor([[1]*len(encodedin) + [0]*(block_size-len(encodedin)-1)], dtype=torch.int)
        #    encodedin.extend([220]*(block_size-len(encodedin)-1))
        #else:
        inputmask = torch.tensor([[1]*len(encodedin)], dtype=torch.int)
        
        context = torch.tensor([encodedin], dtype=torch.long, device=device)
        print(tgt_tokenizer.decode(model.generate(context, gen, inputmask, max_new_tokens=block_size)[0].tolist()))