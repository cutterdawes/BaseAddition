'''
Class to generate addition problems

Output is: a tuple consisting of (sequence, digits of answer, ids of query tokens)

For example 12+34=46 would correspond to ([2,4,?,1,3,?],[6,4],[2,5])
where each entry in the first list is actually a one-hot vector
and "?" is a special token
so the shape of the first element in the tuple is (sequence length) x (number of tokens)
where the number of tokens is the modulus plus 1 (for the "?" token)

Arguments:
-carry_table, a dxd array
-depth: number of digits in each summand, positive integer
-ids: list of integers to use when generating problems; a list of integers in the range [0,d^depth)
(ids can also be None, in which case all possible tuples will be used to generate problems)
(Note: this can be used for making train/test splits: you will probably want to
generate two disjoint lists of integers, use one of them as "ids" for the train dataset and the other for the test dataset)
-debug_mode: if True, will set the total number of problems to a small value, useful for testing new code
-digit_order: 'standard' corresponds to most-significant digit first and 'reversed' is least-significant digit first
-input_format: must be 'onehot'
-interleaved: if True, will interleave input like [2,4,?,1,3,?], if False, will concatenate like [2,1,4,3,?,?]
-ans_at_end: if True and interleaved=True, will put query tokens at the end like [2,4,1,3,?,?]; this argument is ignored when interleaved=False

Recommended values of parameters to start with:
depth: 2 or 3
digit_order: reversed
interleaved: True
ans_at_end: False
'''

import torch
import numpy as np
import sys
sys.path.append('../')
from base import BaseElt

def _tuple_to_int(vals, b):
    #convert (d1,...,dk) to a unique integer, where d_i=0,1,...,b-1
    pow = [1]
    while len(pow)<len(vals):
        pow = [b*pow[0]] + pow
    return sum([n*m for n, m in zip(pow, vals)])

def _interleave_lists(*args):
    #https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
    return [val for pair in zip(*args) for val in pair]

class GroupAddition(torch.utils.data.Dataset):
    def __init__(self, carry_table, depth, ids, debug_mode=False, digit_order='standard', input_format='onehot', interleaved=False, ans_at_end=False):
        self.carry_table = carry_table
        self.depth = depth #number of components in vector
        self.b = len(carry_table)
        self.ids = ids #a list of integers from 0 to b**(depth) indicating valid possibilites for g1 and g2
        #useful to train/test splits (could be None, in which case there is no restriction)
        self.debug_mode = debug_mode
        self.digit_order = digit_order
        assert digit_order in ['standard','reversed']
        self.input_format = input_format
        assert input_format in ['onehot','integer'] #integer representation used for fourier embedding
        #integer in each entry depends on the base; for example consider binary sequence (1,0,1)
        #the integer representation of this would be (4,0,1)
        self.interleaved = interleaved #given two ints (a,b,c),(d,e,f) (where a=most significant digit)
        #if interleaved, will order them like (a,d,?,b,e,?,c,f,?) where ? is answer token
        #otherwise, will order them like (a,b,c,d,e,f,?,?,?)
        self.ans_at_end = ans_at_end #if True, then will show all digits first, with space for answer afterweards
        #if False, will interleave the query tokens in with the digit tokens
        self.len=100 if self.debug_mode else self.b**self.depth if self.ids is None else len(self.ids)

    def __len__(self):
        return self.len
    
    def set_depth(self,depth):
        self.depth = depth

    def __getitem__(self, idx):
        while True:
            v1 = np.random.choice(self.b, size=self.depth)
            if self.ids is None or _tuple_to_int(v1, self.b) in self.ids:
                break
        while True:
            v2 = np.random.choice(self.b, size=self.depth)
            if self.ids is None or _tuple_to_int(v2, self.b) in self.ids:
                break
        g1 = BaseElt(tuple(v1), self.carry_table)
        g2 = BaseElt(tuple(v2), self.carry_table)
        s = list((g1 + g2).vals)
        #only consider cyclic addition
        if len(s) > self.depth:
            s = s[-self.depth:]
        #zero pad
        elif len(s) < self.depth:
            s = [0] * (self.depth - len(s)) + s
        bases = [self.b**i for i in reversed(range(self.depth))] #used for integer representations

        if self.digit_order == 'reversed':
            v1 = v1[::-1]
            v2 = v2[::-1]
            s = s[::-1]
            bases = bases[::-1]

        if self.interleaved:
            inp = _interleave_lists(v1, v2)+[self.b] + [-1] * len(v1) if self.ans_at_end else _interleave_lists(v1, v2, [-1]*len(v1))
            ids = torch.Tensor([i for i, x in enumerate(inp) if x==-1]).long()
            if self.input_format == 'onehot':
                X = torch.zeros((len(inp), self.b + (1 if self.ans_at_end else 0)))
                for i, j in enumerate(inp):
                    if j >= 0:
                        X[i, j] = 1
            else:
                raise ValueError('')
            return X, torch.Tensor(s).long(), ids
        else:
            inp = np.array(list(v1) + [self.b] + list(v2))
            if self.input_format == 'onehot':
                #convert to one-hots, with empty entries for filling in the answer
                X=torch.zeros((len(inp) + len(s), self.b + 1))
                for i, j in enumerate(inp):
                    X[i, j] = 1
            else:
                raise ValueError('')
            #positions with placeholders
            ids = torch.arange(len(inp), len(X))
            return X, torch.Tensor(s).long(), ids