# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

def load_corpus(file_name):
    word_to_ix = {}
    ix_to_word = {}
    word_cnt = 0

    with open(file_name) as fopen:
        for line in fopen:
            for word in line:
                if word not in word_to_ix:
                    word_to_ix[word] = word_cnt
                    ix_to_word[word_cnt] = word
                    word_cnt += 1
    word_to_ix['<unk>'] = word_cnt
    ix_to_word[word_cnt] = '<unk>'
    word_cnt += 1
    return word_to_ix, ix_to_word

def char_tensor(string, word_to_ix):
    tensor = torch.zeros(len(string)).long()
    for idx in range(len(string)):
        tensor[idx] = word_to_ix.get(string[idx], word_to_ix['<unk>'])
    return Variable(tensor)

def train_data_generator(file_name, word_to_ix, chunk_size=200, batch_size=128):
    batch_data = []
    chunk = ""
    with open(file_name) as fopen:
        for line in fopen:
            for word in line.strip():
                if len(chunk) < chunk_size:
                    chunk += word
                else:
                    #yield chunk
                    batch_data.append((char_tensor(chunk[:-1], word_to_ix), char_tensor(chunk[1:], word_to_ix)))
                    chunk = ""
                    if len(batch_data) == batch_size:
                        yield batch_data
                        batch_data = []

def train_data_generator_shuffle(file_name, word_to_ix, chunk_size=200, batch_size=128, shuffle_buf_size=2056):
    import random
    buf = []
    for batch_data in train_data_generator(file_name, word_to_ix, chunk_size, batch_size):
        buf.extend(batch_data)
        if len(buf) >= shuffle_buf_size:
            random.shuffle(buf)
            for _ in buf:
                yield _
            buf = []

def evaluate(char_rnn, word_to_ix, prime_str='一', predict_len=50, temperature=0.8):
    hidden = char_rnn.init_hidden()
    prime_input = char_tensor(prime_str, word_to_ix)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = char_rnn(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = char_rnn(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = ix_to_word[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char, word_to_ix)

    return predicted

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    # Data Preprocess Phrase
    corpus_file = './sanguo.txt'
    chunk_size = 200
    word_to_ix, ix_to_word = load_corpus(corpus_file)
    print('Corpus contains %d words' % (len(word_to_ix)))
    #h = train_data_generator_shuffle(corpus_file, word_to_ix, 5, 2, 100)


    # Train Phrase
    print_every = 500
    plot_every = 100
    hidden_size = 100
    n_layers = 1
    lr = 0.005
    char_rnn = CharRNN(len(word_to_ix), hidden_size, len(word_to_ix), n_layers)
    char_rnn_optimizer = torch.optim.Adam(char_rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    cumulative_loss = 0.0
    
    # TODO: train on batch, use module data loader
    step_cnt = 0
    for batch_data in train_data_generator_shuffle(corpus_file, word_to_ix, 128, 16, 1024):
        for _ in batch_data:
            step_cnt += 1
            inp, target = _[0], _[1]
            hidden = char_rnn.init_hidden()
            char_rnn.zero_grad()
            loss = 0
            for c in range(len(inp)):
                output, hidden = char_rnn(inp[c], hidden)
                loss += criterion(output, target[c])
            loss.backward()
            char_rnn_optimizer.step()
            cumulative_loss += (loss.data[0]/len(inp))

            if step_cnt % print_every == 0:
                print('[%s (#%d) %.4f]' % (time_since(start), step_cnt, cumulative_loss/step_cnt))
                print(evaluate(char_rnn, word_to_ix, '一日', 50), '\n')



    # Eval Phrase
    #print(evaluate(char_rnn, word_to_ix, prime_str='回'))
