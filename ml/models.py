import torch
from torch import nn


class RNN(nn.Module):
    '''simple recurrent neural network model'''

    def __init__(self, b, layers):
        super().__init__()
        '''initialize model'''
        self.b = b
        self.layers = layers        
        self.rnn = nn.RNN(b, b, layers, batch_first=True)
        self.linear = nn.Linear(b, b)

    def forward(self, x):
        '''basic forward-pass'''
        x_out, _ = self.rnn(x)
        x_out = self.linear(x_out)
        return x_out
        
    def logits(self, x, ids):
        '''return (un-normalized) probability logits for sum'''
        x_out = self(x)
        x_out = x_out.squeeze()
        if x_out.dim() == 3:
            x_out_and_ids = zip(torch.unbind(x_out), torch.unbind(ids))
            s_out = torch.stack([x_out[ids,:] for x_out, ids in x_out_and_ids])
        elif x_out.dim() == 2:
            s_out = x_out[ids,:]
        else:
            raise ValueError('invalid x_out dim')
        return s_out.squeeze()
    
    def predict(self, x, ids):
        '''return predicted sum'''
        s_out = self.logits(x, ids)
        s_out = torch.argmax(s_out, -1)
        return s_out
    

class GRU(nn.Module):
    '''simple gated recurrent unit model'''

    def __init__(self, b, layers):
        super().__init__()
        '''initialize model'''
        self.b = b
        self.layers = layers        
        self.gru = nn.GRU(b, b, layers, batch_first=True)
        self.linear = nn.Linear(b, b)

    def forward(self, x):
        '''basic forward-pass'''
        x_out, _ = self.gru(x)
        x_out = self.linear(x_out)
        return x_out
        
    def logits(self, x, ids):
        '''return (un-normalized) probability logits for sum'''
        x_out = self(x)
        x_out = x_out.squeeze()
        if x_out.dim() == 3:
            x_out_and_ids = zip(torch.unbind(x_out), torch.unbind(ids))
            s_out = torch.stack([x_out[ids,:] for x_out, ids in x_out_and_ids])
        elif x_out.dim() == 2:
            s_out = x_out[ids,:]
        else:
            raise ValueError('invalid x_out dim')
        return s_out.squeeze()
    
    def predict(self, x, ids):
        '''return predicted sum'''
        s_out = self.logits(x, ids)
        s_out = torch.argmax(s_out, -1)
        return s_out


class LSTM(nn.Module):
    '''small LSTM-based addition model'''

    def __init__(self, b, layers):
        super().__init__()
        '''initialize model'''
        self.b = b
        self.layers = layers
        self.lstm = nn.LSTM(b, b, layers, batch_first=True)
        self.linear = nn.Linear(b, b)

    def forward(self, x):
        '''basic forward-pass'''
        x_out, _ = self.lstm(x)
        x_out = self.linear(x_out)
        return x_out
    
    def logits(self, x, ids):
        '''return (un-normalized) probability logits for sum'''
        x_out = self(x)
        x_out = x_out.squeeze()
        if x_out.dim() == 3:
            x_out_and_ids = zip(torch.unbind(x_out), torch.unbind(ids))
            s_out = torch.stack([x_out[ids,:] for x_out, ids in x_out_and_ids])
        elif x_out.dim() == 2:
            s_out = x_out[ids,:]
        else:
            raise ValueError('invalid x_out dim')
        return s_out.squeeze()
    
    def predict(self, x, ids):
        '''return predicted sum'''
        s_out = self.logits(x, ids)
        s_out = torch.argmax(s_out, -1)
        return s_out