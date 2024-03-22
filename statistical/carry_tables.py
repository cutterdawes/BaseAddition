# import packages
import argparse
from mpi4py import MPI
import pickle
from itertools import product
import numpy as np
import sys
sys.path.append('../')
import fn

def pickle_tables(tables, args):
    directory = '/homes/cdawes/Repo/pickles/bases' if (args.directory is None) else args.directory
    with open(f'{directory}/tables{args.base}.pickle', 'wb') as file:
        pickle.dump(tables, file)
    if not args.parallel:
        print(f'Function executed successfully.\nOutput saved to {directory}/tables{args.base}.pickle')

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
    scattered_tables = fn.construct_tables(args.base, rank=rank, size=size)

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
        tables = fn.construct_tables(args.base)
        pickle_tables(tables, args)

if __name__ == '__main__':
    main()