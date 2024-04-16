import random
import torch
import math
from addition_dataset import GroupAddition


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


def prepare(b, depth, table, batch_size=16, split_type='interpolate', split_ratio=0.9, split_depth=-1):
    '''return training and testing dataloader objects for learning addition'''
    
    # get indices of training and testing data
    N = b**depth
    ids = _get_ids(b, depth, split_type, split_ratio, split_depth)
    heldout_ids = set(range(N)) - set(ids)
    
    # create training dataset and dataloader
    training_dataset = GroupAddition(table, depth, ids=ids, interleaved=True, digit_order='reversed')
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    
    # create testing dataset and dataloader
    testing_dataset = GroupAddition(table, depth, ids=heldout_ids, interleaved=True, digit_order='reversed')
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset)

    return training_dataloader, testing_dataloader