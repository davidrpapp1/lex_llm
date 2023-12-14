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


# Enable/disable training mode and the reprocessing of training data located in reservoir
training_mode = True
reprocess_data = False
small_batch_validation_output = False # Displays snapshot of tensor indices during vectorisation process


# Miscellaneous parameters
MAX_LENGTH = 10 # Maximum sentence length to keep from training data
MIN_COUNT = 3 # Keep any words from training data that show up equal to or more than MIN_COUNT times


# Configure models
model_name = 'lex_llm' # For file saving label
attn_model = 'dot' # dot, general, or concat
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
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

# Function to trim rarely used words
def trim_rare_words(voc, pairs, MIN_COUNT):
    
    voc.trim(MIN_COUNT)
    
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
            
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input and output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    
    return keep_pairs

# Trim voc and pairs
pairs = trim_rare_words(voc, pairs, MIN_COUNT)


# Each word has an index, which is a unique i.d assigned as the order in which it shows up in the training data, starting from 1
def indices_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [eos_token]

# Function to transpose voc index matrix, and if a sentence has < MAX_LENGTH words, fill the rest of that row in the tensor with 0's
# Transposition is done to be able to index batches W.R.T timesteps, and not W.R.T sentences
def zero_padding(l, fillvalue=pad_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Function to construct a binary matrix to show padding locations
def binary_matrix(l, value=pad_token):
    
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == pad_token:
                m[i].append(0)
            else:
                m[i].append(1)
                
    return m

# Returns padded input sequence tensor and lengths of sequences
def input_tensor(l, voc):
    
    indices_batch = [indices_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indices) for indices in indices_batch])
    pad_list = zero_padding(indices_batch)
    pad_tensor = torch.LongTensor(pad_list)
    
    return pad_tensor, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_tensor(l, voc):
    
    indices_batch = [indices_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indices) for indices in indices_batch])
    pad_list = zero_padding(indices_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_tensor = torch.LongTensor(pad_list)
    
    return pad_tensor, mask, max_target_len

# Returns all items for a given batch of pairs
def process_batch(voc, pair_batch):
    
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        
    input, lengths = input_tensor(input_batch, voc)
    output, mask, max_target_len = output_tensor(output_batch, voc)
    
    return input, lengths, output, mask, max_target_len


# Small batch validation
if(small_batch_validation_output == True):
    small_batch_size = 5
    batches = process_batch(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)



# EncoderRNN class
class EncoderRNN(nn.Module):
    
    # Initialise parameters
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialise bidirectional GRU
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)


    def forward(self, input_seq, input_lengths, hidden=None):
        
        # Convert word indices to embeddings
        embedded = self.embedding(input_seq)
        
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        
        # Return output and final hidden state
        return outputs, hidden
    
    
# Attention layer
class Attn(nn.Module):
    
    def __init__(self, method, hidden_size):
        
        # Allow for different attention scoring mechanisms - methods in Luong et al.
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method, please choose from: dot, general, or concat")
        
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    # Calculate the attention scores based on the chosen method
    def forward(self, hidden, encoder_outputs):
        
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
# Attention decoder class
class LuongAttnDecoderRNN(nn.Module):
    
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate weighted context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        # Return output and final hidden state
        return output, hidden
    

# Calculate loss based on average negative log likelihood
def masked_loss(input, target, mask):
    
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    
    return loss, nTotal.item()


# Function to train the model
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu") # Apparently lengths for RNN should always be on the CPU...

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[sos_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing in this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        
        for t in range(max_target_len):
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            
            # Calculate and accumulate loss
            mask_loss, nTotal = masked_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            
    else: # No teacher forcing
        
        for t in range(max_target_len):
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
           
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            # Calculate and accumulate loss
            mask_loss, nTotal = masked_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def training_iterations(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, 
               save_every, clip, corpus_name, load_file_name):

    # Load batches for each iteration
    training_batches = [process_batch(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

    # Initializations
    print('Initializing')
    start_iteration = 1
    print_loss = 0
    if load_file_name:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training")
    for iteration in range(start_iteration, n_iteration + 1):
        
        training_batch = training_batches[iteration - 1]
        
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
            

# Greedy decoding class
class GreedySearchDecoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        
        # Initialize decoder input with sos_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * sos_token
        
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            
        # Return collections of word tokens and scores
        return all_tokens, all_scores
    
    
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    
    # Transform words to indices
    indices_batch = [indices_from_sentence(voc, sentence)]
    
    # Create lengths tensor
    lengths = torch.tensor([len(indices) for indices in indices_batch])
    
    # Transpose dimensions of batch for compatibility
    input_batch = torch.LongTensor(indices_batch).transpose(0, 1)
    
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    # Decode words from indices
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc):
    
    input_sentence = ''
    
    while(1):
        
        try:
            # Get input sentence
            input_sentence = input('> ')
            
            # Exit prompt
            if input_sentence == 'exit': 
                print('Exiting...')
                break
            
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Lex:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word")
            

# Set checkpoint to load from, or None if training
if training_mode == True: 
    load_file_name = None

if training_mode != True:
    load_file_name = os.path.join(save_dir, model_name, corpus_name,
                        '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                        '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a load_file_name is provided
if training_mode != True:
    
    if load_file_name:
        
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_file_name)
        
        #checkpoint = torch.load(load_file_name, map_location=torch.device('cpu')) # If loading a model trained on GPU to CPU
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder')

# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if load_file_name:
    embedding.load_state_dict(embedding_sd)
    
# Initialize encoder & decoder models
encoder = EncoderRNN(attn_model, hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if load_file_name:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Model built')


if training_mode == True:

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if load_file_name:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have CUDA, configure parallelisation
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Training...")
    training_iterations(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
            embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
            print_every, save_every, clip, corpus_name, load_file_name)


if training_mode != True:
    
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Request user input
    evaluate_input(encoder, decoder, searcher, voc)