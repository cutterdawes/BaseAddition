# import packages
import argparse
from mpi4py import MPI
import pickle
import fn

def pickle_tables(tables, args):
    directory = '/scratch/network/cdawes' if (args.directory is None) else args.directory
    with open(f'{directory}/tables{args.b}.pickle', 'wb') as file:
        pickle.dump(tables, file)
    if args.c is None:
        print(f'Function executed successfully.\nOutput saved to {directory}/tables{args.b}.pickle')

def parallel_case(args):
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create and scatter candidate cocycles to workers
    cs = list(product(*[range(args.b)]*(args.b-1)))
    scattered_cs = np.empty_like(cs)
    comm.Scatter(cs, scattered_cs, root=0)

    # check candidate cocycles on each worker's portion
    scattered_tables = fn.construct_tables(args.b, cs=scattered_cs)

    # gather tables from all workers
    gathered_tables = comm.gather(scattered_tables, root=0)

    # consolidate all tables on root process, then pickle
    if rank == 0:
        tables = {}
        for some_tables in gathered_tables:
            tables.update(some_tables)
        pickle_tables(tables, args)

def main(): 
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True, help='Specified base')
    parser.add_argument('-c', '--cores', type=int, required=False, help='Number of cores for parallel processing')
    parser.add_argument('--directory', type=str, required=False, help='directory of pickled tables')
    args = parser.parse_args()
    
    # compute carry tables
    if args.c is None:
        # serial case
        tables = fn.construct_tables(args.b)
        pickle_tables(tables, args)
    else:
        # parallel case
        parallel_case(args)
    
if __name__ == '__main__':
    main()