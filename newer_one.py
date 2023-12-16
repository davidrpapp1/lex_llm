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


# Enable/disable training mode and the reprocessing of training data located in reservoir
training_mode = True
reprocess_data = False
small_batch_validation_output = True # Displays snapshot of tensor indices during vectorisation process
train_val_split = 0.9 # fraction of data used for training
encoding = tiktoken.get_encoding("cl100k_base")
vocab_size=100256

# Miscellaneous parameters
MAX_LENGTH = 10 # Maximum sentence length to keep from training data
MIN_COUNT = 1 # Keep any words from training data that show up equal to or more than MIN_COUNT times


# Configure models
model_name = 'lex_llm' # For file saving label
eval_interval = 10
eval_iters = 5
n_embed = 64
n_head = 6
n_layer = 6
dropout = 0.2
batch_size = 64
block_size = 19
max_iters = 30


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
n_iteration = 100
checkpoint_iter = 4000 # If using already trained model, set to total iterations for that training data
print_every = 1
save_every = 500


# Use CUDA if installed on current system, otherwise use CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Declare folder name within reservoir
corpus_name = "movie-corpus"
corpus = os.path.join("training", corpus_name)

# Define path to new file
datafile = os.path.join(corpus, "formatted_lines.txt")

# These functions enable re-processing of data
if reprocess_data == True:

    # Splits each line of the file to create lines and conversations
    def load_lines(file_name):
        
        lines = {}
        conversations = {}
        with open(file_name, 'r', encoding='iso-8859-1') as f:
            
            for line in f:
                
                # Extract fields for line object
                line_json = json.loads(line)
                line_obj = {}
                line_obj["lineID"] = line_json["id"]
                line_obj["characterID"] = line_json["speaker"]
                line_obj["text"] = line_json["text"]
                lines[line_obj['lineID']] = line_obj

                # Extract fields for conversation object
                if line_json["conversation_id"] not in conversations:
                    convObj = {}
                    convObj["conversationID"] = line_json["conversation_id"]
                    convObj["movieID"] = line_json["meta"]["movie_id"]
                    convObj["lines"] = [line_obj]
                else:
                    convObj = conversations[line_json["conversation_id"]]
                    convObj["lines"].insert(0, line_obj)
                conversations[convObj["conversationID"]] = convObj

        return lines, conversations


    # Extract question and answers from conversations
    def extract_q_a(conversations):
        
        qa_pairs = []
        
        for conversation in conversations.values():
            
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                q_line = conversation["lines"][i]["text"].strip()
                a_line = conversation["lines"][i+1]["text"].strip()
                
                # Filter wrong samples (if one of the lists is empty)
                if q_line and a_line:
                    qa_pairs.append([q_line, a_line])
                    
        return qa_pairs

    # Unescape the delimiter
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines and conversations dictionary
    lines = {}
    conversations = {}
    
    # Load lines and conversations
    print("Processing training data into lines and conversations")
    lines, conversations = load_lines(os.path.join(corpus, "utterances.jsonl"))

    # Write new csv file
    print("Writing into CSV file")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_q_a(conversations):
            writer.writerow(pair)

# Default word token ennumeration
pad_token = 0  # Used for padding short sentences
sos_token = 1  # Start-of-sentence token
eos_token = 2  # End-of-sentence token

# Vocabulary class
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {pad_token: "PAD", sos_token: "SOS", eos_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_word_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_word_count:
                keep_words.append(k)

        # Print ratio of kept words
        print('\nKept words: {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {pad_token: "PAD", sos_token: "SOS", eos_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.add_word(word)

# Turn a Unicode string to plain ASCII,
def unicode_to_ascii(s):
    
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    
    return s

# Read question/answer pairs and return a Voc object
def read_vocabulary(datafile, corpus_name):
    
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    
    return voc, pairs

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(p):
    
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using the filter_pair condition
def filter_pairs(pairs):
    
    return [pair for pair in pairs if filter_pair(pair)]

# Using the functions defined above, return a populated Voc object and pairs list
def load_prepare_data(corpus, corpus_name, datafile, save_dir):
    
    print("Loading training data")
    voc, pairs = read_vocabulary(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.num_words)
    
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir)

# Print some pairs to validate
print("\nSample pairs:")
for pair in pairs[:10]:
    print(pair)


datain=[]
dataout=[]
for q in range(len(pairs)):
    datain_e = encoding.encode(pairs[q][0])
    dataout_e = encoding.encode(pairs[q][1])
    if len(datain_e) < block_size:
        datain_e.extend([220]*(block_size - len(datain_e)))
    if len(dataout_e) < block_size:
        dataout_e.extend([220]*(block_size - len(dataout_e)))

    datain.append(torch.tensor(datain_e, dtype=torch.long))
    dataout.append(torch.tensor(dataout_e, dtype=torch.long))


n=int(train_val_split*len(datain))
datain_t = datain[:n]
datain_v = datain[n:]
dataout_t = dataout[:n]
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
    xin, xout, yin, yout = xin.to(device), xout.to(device), yin.to(device), yout.to(device)
    return xin, xout, yin, yout

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xin, Xout, Yin, Yout = get_batch(split)
            logits, loss = model(Xin, Xout, targets=Yout)
            losses[k] = loss.item()
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

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
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

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out




class NoMaskHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(enc_out)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
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

    def forward(self, x, enc_out):
        out = torch.cat([h(x, enc_out) for h in self.heads], dim=-1)
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

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
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

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
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

    def forward(self, x, enc_out):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.lnc(x), enc_out)
        x = x + self.ffwd(self.ln2(x))
        return x
    
class LangMod(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.encoderblocks = nn.Sequential(*[EncoderBlock(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #can be RMS norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, enc_out, idx, targets=None):
        B,T = idx.shape
        Be, Te = enc_out.shape

        tok_emb_inp = self.token_embedding_table(enc_out)
        pos_emb_inp = self.position_embedding_table(torch.arange(Te, device=device))
        x_input = tok_emb_inp + pos_emb_inp

        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb # (B,T,C)

        x_enc = self.encoderblocks(x_input)
        x = self.blocks(x, x_enc) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inp, idx, max_new_tokens):
        for _ in range((max_new_tokens)):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(inp, idx_cond)
            logits = logits[:, -1, :] # last time step (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
model = LangMod()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xbi, xbo, ybi, ybo = get_batch("train")

    logits, loss = model(xbi, xbo, targets=ybo)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

on = True
while on:
    q = input()
    if q == "quit":
        on = False
    else:
        gen = torch.zeros((1,1), dtype=torch.long, device=device)
        context = torch.tensor([encoding.encode(q)], dtype=torch.long, device=device)
        print(encoding.decode(m.generate(context, gen, max_new_tokens=10)[0].tolist()))

