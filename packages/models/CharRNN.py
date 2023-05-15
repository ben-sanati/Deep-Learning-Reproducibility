import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode(arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        
        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        
        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        
        return one_hot

class CharRNN(nn.Module):
    def __init__(self, tokens, batch_size, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        # Creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## Define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        
        ## Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## Define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))
        
        # Initialize the weights
        self.init_weights()

        weight = next(self.parameters()).data
        
        self.hc = (weight.new(self.n_layers, int(batch_size / torch.cuda.device_count()), self.n_hidden).zero_(), weight.new(self.n_layers, int(batch_size / torch.cuda.device_count()), self.n_hidden).zero_())
    
    def forward(self, x):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hc`. '''
        
        ## Get x, and the new hidden state (h, c) from the lstm
        hc = tuple([each.data.to(x.device) for each in self.hc])
        x, (h, c) = self.lstm(x, hc)
        
        ## Ppass x through the dropout layer
        x = self.dropout(x)
        
        # Stack up LSTM outputs using view
        x = x.reshape(x.size()[0]*x.size()[1], self.n_hidden)
        
        ## Put x through the fully-connected layer
        x = self.fc(x)
        
        # Return x and the hidden state (h, c)
        self.hc = (h, c)
        return x, (h, c)
    
    
    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.
        
            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()
        
        if h is None:
            h = self.init_hidden(1)
        
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        
        inputs = torch.from_numpy(x)
        
        if cuda:
            inputs = inputs.cuda()
        
        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        _, prediction = torch.max(out, dim=1)
        char_prediction = self.model.int2char[prediction]
            
        return self.int2char[char_prediction], h
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
