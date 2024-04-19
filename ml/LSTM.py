import torch
from torch import nn


class LSTM(nn.Module):
    '''small LSTM-based addition model'''

    def __init__(self, b, layers):
        super().__init__()
        '''initialize model'''
        self.b = b
        self.layers = layers
        self.lstm = nn.LSTM(b, b, layers, batch_first=True)
        self.linear = nn.Linear(b, b)

    def forward(self, X):
        '''basic forward-pass'''
        X_out, _ = self.lstm(X)
        X_out = self.linear(X_out)
        return X_out
    
    def logits(self, X, ids):
        '''return (un-normalized) probability logits for sum'''
        X_out = self(X)
        X_out = X_out.squeeze()
        if X_out.dim() == 3:
            X_out_and_ids = zip(torch.unbind(X_out), torch.unbind(ids))
            s_out = torch.stack([X_out[ids,:] for X_out, ids in X_out_and_ids])
        elif X_out.dim() == 2:
            s_out = X_out[ids,:]
        else:
            raise ValueError('invalid X_out dim')
        return s_out.squeeze()
    
    def predict(self, X, ids):
        '''return predicted sum'''
        s_out = self.logits(X, ids)
        s_out = torch.argmax(s_out, -1)
        return s_out