import argparse
import pickle
import numpy as np
import addition_data
import torch
import addition_eval
from LSTM import LSTM


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True, help='Specified base')
    parser.add_argument('-d', '--directory', type=str, required=False, help='directory of pickled output')
    parser.add_argument('-n', '--num_workers', type=int, required=False, help='number of CPU workers for data preparation')
    args = parser.parse_args()

    # get carry tables
    with open('../pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]
    
    # specify torch device (set to GPU if available), set number of CPU workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 1 if (args.num_workers is None) else args.num_workers

    # get table
    dc = ((0,)*args.base,)*args.base
    table = tables[dc]

    # initialize model and dataloaders
    model = LSTM(args.base, 1).to(device)
    training_dataloader, testing_dataloader = addition_data.prepare(args.base, 6, table, split_type='OOD', split_depth=3, sample=True, num_workers=num_workers)

    # evaluate model and store output
    __ = addition_eval.eval(model, training_dataloader, testing_dataloader, device, num_passes=500, print_loss_and_acc=True)


if __name__ == '__main__':
    main()