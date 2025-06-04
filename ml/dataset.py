'''
Class to generate addition problems

Output is: a tuple consisting of (sequence, digits of answer, ids of query tokens)

For example 12+34=46 would correspond to ([2,4,?,1,3,?],[6,4],[2,5])
where each entry in the first list is actually a one-hot vector
and "?" is a special token
so the shape of the first element in the tuple is (sequence length) x (number of tokens)
where the number of tokens is the modulus plus 1 (for the "?" token)

Arguments:
- carry_table, a dxd array
- depth: number of digits in each summand, positive integer
- ids: list of integers to use when generating problems; a list of integers in the range [0,d^depth)
(ids can also be None, in which case all possible tuples will be used to generate problems)
(Note: this can be used for making train/test splits: you will probably want to
generate two disjoint lists of integers, use one of them as "ids" for the train dataset and the other for the test dataset)
- debug_mode: if True, will set the total number of problems to a small value, useful for testing new code
- digit_order: 'standard' corresponds to most-significant digit first and 'reversed' is least-significant digit first
- input_format: must be 'onehot'
- interleaved: if True, will interleave input like [2,4,?,1,3,?], if False, will concatenate like [2,1,4,3,?,?]
- ans_at_end: if True and interleaved=True, will put query tokens at the end like [2,4,1,3,?,?]; this argument is ignored when interleaved=False

Recommended values of parameters to start with:
depth: 2 or 3
digit_order: reversed
interleaved: True
ans_at_end: False
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import gaussian_blur
from typing import Tuple, List, Union
import numpy as np
import random
import math
import sys
sys.path.append('../')
from math.base_rep import CarryTable, BaseElt


def _int_to_tuple(n, b):
    nb = ()
    while n:
        nj = n % b
        nb += (nj,)
        n //= b
    return nb[::-1]


def _gaussian_envelope(X: torch.tensor, u: int = 1):

    # get base, kernel size, and padding
    b = X.shape[-1]
    size = b if (b % 2 == 1) else b-1
    pad = size // 2

    # determine ordering by specified generator
    ordering = torch.tensor([(k*u)%b for k in range(b)])
    ordering_inverse = torch.argsort(ordering)
    
    # re-order X, pad and blur X, undo re-ordering
    X_reordered = X[:,ordering]
    X_pad = F.pad(X_reordered, (pad, pad), mode='circular')
    X_blurred = gaussian_blur(X_pad.unsqueeze(0), kernel_size=(size, 1)).squeeze()[:,pad:b+pad]
    X_blurred = X_blurred[:,ordering_inverse]

    return X_blurred


def _interleave_lists(*args):
    # https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
    return [val for pair in zip(*args) for val in pair]


def _get_ids(b, depth, split_type, split_ratio=0.9, split_depth=-1):
    '''return ids corresponding to training dataloader depending on generalization type'''

    # check inputs, initialize variables
    assert split_type in ['interpolate', 'OOD'], 'invalid type'
    N = b**depth

    # training ids if testing interpolation generalization
    if split_type == 'interpolate':
        assert (0 < split_ratio < 1), 'invalid split'
        ids = random.sample(range(N), math.ceil(split_ratio * N))

    # training ids if testing O.O.D. generalization
    elif split_type == 'OOD':
        assert (0 < split_depth <= depth), 'invalid sample_depth'
        ids = list(range(b**split_depth))

    return ids


class BaseAddition(Dataset):
    def __init__(
        self,
        carry_table: Union[np.ndarray, CarryTable],
        depth: int,
        ids: List = None,
        debug_mode: bool = False,
        digit_order: str = 'reversed',
        input_format: str = 'onehot',
        interleaved: bool = True,
        ans_at_end: bool = False,
        semanticity: bool = False,
        unit: int = 1
    ):
        self.carry_table = carry_table
        self.depth = depth # number of components in vector
        self.b = len(carry_table)
        self.ids = ids if (ids is not None) else np.arange(self.b**self.depth)
        # a list of integers from 0 to b**(depth) indicating valid possibilites for g1 and g2
        #useful to train/test splits
        self.debug_mode = debug_mode
        self.digit_order = digit_order
        assert digit_order in ['standard','reversed']
        self.input_format = input_format
        assert input_format in ['onehot','integer'] # integer representation used for fourier embedding
        # integer in each entry depends on the base; for example consider binary sequence (1,0,1)
        # the integer representation of this would be (4,0,1)
        self.interleaved = interleaved # given two ints (a,b,c),(d,e,f) (where a=most significant digit)
        # if interleaved, will order them like (a,d,?,b,e,?,c,f,?) where ? is answer token
        # otherwise, will order them like (a,b,c,d,e,f,?,?,?)
        self.ans_at_end = ans_at_end # if True, then will show all digits first, with space for answer afterweards
        # if False, will interleave the query tokens in with the digit tokens
        self.semanticity = semanticity # if True, enforces semantic digit ordering by convolving one-hot vectors 
        # with Gaussian depending on ordering specified by unit argument
        self.unit = unit # unit used as generator (e.g., if 3 for base 5, then ordering is [0, 3, 1, 4, 2])
        assert unit in range(self.b)
        self.len=100 if self.debug_mode else self.b**self.depth if self.ids is None else len(self.ids)

    def __len__(self):
        return self.len
    
    def set_depth(self, depth: int):
        self.depth = depth

    def __getitem__(self, idx: int):

        # get numbers to add
        n = _int_to_tuple(np.random.choice(self.ids), self.b)
        m = _int_to_tuple(np.random.choice(self.ids), self.b)
        if len(n) != self.depth:
            zp = self.depth - len(n)
            n = (0,) * zp + n
        if len(m) != self.depth:
            zp = self.depth - len(m)
            m = (0,) * zp + m

        # get sum of n and m
        n = BaseElt(tuple(n), self.carry_table)
        m = BaseElt(tuple(m), self.carry_table)
        s = n + m

        # convert back to tuples
        s = s.vals
        n = n.vals
        m = m.vals

        bases = [self.b**i for i in reversed(range(self.depth))] # used for integer representations

        if self.digit_order == 'reversed':
            n = n[::-1]
            m = m[::-1]
            s = s[::-1]
            bases = bases[::-1]

        if self.interleaved:
            inp = _interleave_lists(n, m)+[self.b] + [-1] * len(n) if self.ans_at_end else _interleave_lists(n, m, [-1]*len(n))
            ids = torch.Tensor([i for i, x in enumerate(inp) if x==-1]).long()
            if self.input_format == 'onehot':
                X = torch.zeros((len(inp), self.b + (1 if self.ans_at_end else 0)))
                for i, j in enumerate(inp):
                    if j >= 0:
                        X[i, j] = 1
                s = torch.Tensor(s).long()
            else:
                raise ValueError('')
        else:
            inp = np.array(list(n) + [self.b] + list(m))
            if self.input_format == 'onehot':
                #convert to one-hots, with empty entries for filling in the answer
                X=torch.zeros((len(inp) + len(s), self.b + 1))
                for i, j in enumerate(inp):
                    X[i, j] = 1
                s = torch.Tensor(s).long()
            else:
                raise ValueError('')
            
        if self.semanticity:
            X = _gaussian_envelope(X, self.unit)
            # s = _gaussian_envelope(X, self.unit)

        return X, s, ids


def prepare(
    b: int,
    depth: int,
    table: Union[np.ndarray, CarryTable],
    semanticity: bool = False,
    unit: int = 1,
    batch_size: int = 16,
    split_type: str = 'interpolate',
    split_ratio: float = 0.9,
    split_depth: int = -1,
    sample: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    '''return training and testing dataloader objects for learning addition'''
    
    # get indices of training and testing data
    N = b**depth
    ids = _get_ids(b, depth, split_type, split_ratio, split_depth)
    heldout_ids = set(range(N)) - set(ids)
    if sample:
        heldout_ids = random.sample(heldout_ids, len(ids))
    
    # create training dataset and dataloader
    training_dataset = BaseAddition(table, depth, ids=ids, semanticity=semanticity, unit=unit)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # create testing dataset and dataloader
    testing_dataset = BaseAddition(table, depth, ids=heldout_ids, semanticity=semanticity, unit=unit)
    testing_dataloader = DataLoader(testing_dataset, shuffle=True, num_workers=num_workers)

    return training_dataloader, testing_dataloader