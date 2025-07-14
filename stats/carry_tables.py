import argparse
import pickle
import os
import numpy as np
from mpi4py import MPI

import utils


def pickle_tables(tables, args):
    # create directory if it does not exist
    directory = 'pickles/carry_tables' if (args.directory is None) else args.directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    # load all_tables if it exists, otherwise create an empty dictionary
    pickle_path = f'{directory}/all_tables.pickle'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            all_tables = pickle.load(f)
    else:
        all_tables = {}
    # update all_tables with the new tables
    if args.base in all_tables:
        all_tables[args.base].update(tables)
    else:
        all_tables[args.base] = tables
    # pickle all_tables
    with open(f'{pickle_path}', 'wb') as file:
        pickle.dump(all_tables, file)
    if not args.parallel:
        print(f'Function executed successfully.\nOutput saved to {pickle_path}.pickle')


def consolidate_tables(tables):
    consolidated_tables = {}
    for c, table in tables.items():
        added = False
        for o_c, o_table in consolidated_tables.items():
            if np.array_equal(table, o_table):
                consolidated_tables.pop(o_c)
                o_c += c
                consolidated_tables[o_c] = table
                added = True
                break
        if not added:
            consolidated_tables[c] = table
    return consolidated_tables


def parallel_case(args):
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # check candidate cocycles on each worker's portion
    scattered_tables = utils.construct_tables(args.base, rank=rank, size=size)

    # gather tables from all workers
    gathered_tables = comm.gather(scattered_tables, root=0)

    # consolidate all tables on root process, then pickle
    if rank == 0:
        tables = {}
        for some_tables in gathered_tables:
            tables.update(some_tables)
        tables = consolidate_tables(tables)
        pickle_tables(tables, args)


def main(): 
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True, help='Specified base')
    parser.add_argument('-p', '--parallel', action='store_true', help='Specify if processing in parallel')
    parser.add_argument('-d', '--directory', type=str, required=False, help='directory of pickled tables')
    args = parser.parse_args()
    
    # compute carry tables
    if args.parallel:
        # parallel case
        parallel_case(args)
    else:
        # serial case
        tables = utils.construct_tables(args.base)
        pickle_tables(tables, args)


if __name__ == '__main__':
    main()