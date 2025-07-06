import argparse
from itertools import product, combinations
import random
import pickle
from mpi4py import MPI

from base_rep import BaseElt


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, required=True)
    parser.add_argument('-p', '--parallel', action='store_true', default=False,
                        help='run in parallel with MPI (default: False)')
    args = parser.parse_args()

    # initialize MPI if parallel
    if args.parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    # initialize associativity, all tables
    all_associativity = {}
    with open('pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)

    # iterate through bases
    for b in range(3, 6):
        print(f'base {b}...')

        # initialize dictionary
        associativity = {}

        # iterate through tables
        for i, (dc, table) in enumerate(all_tables[b].items()):
            if args.parallel and (i % size != rank):
                continue
            
            # iterate through depths
            assoc_vs_depth = []
            for depth in range(2, args.depth + 1):
                tuples = list(product(*[range(b)]*depth))
                if depth > 3:
                    tuples = random.sample(tuples, b**3)
                triplets = list(combinations(tuples, 3))
                if len(triplets) > 1000:
                    triplets = random.sample(triplets, 1000)
                num_assoc = 0
                for (n, m, p) in triplets:
            
                    # convert to group elements
                    n = BaseElt(n, table)
                    m = BaseElt(m, table)
                    p = BaseElt(p, table)
            
                    # check associativity
                    s1 = (n + m) + p
                    s2 = n + (m + p)
                    is_assoc = s1.vals == s2.vals
                    if is_assoc:
                        num_assoc += 1

                # calculate associativity
                assoc = num_assoc / len(triplets)
                assoc_vs_depth.append(assoc)
        
            # add fraction of associativity to dictionary
            associativity[dc] = assoc_vs_depth

        # gather associativity dict if parallel
        if args.parallel:
            gathered_associativity = comm.gather(associativity, root=0)
            if rank == 0:
                associativity = {}
                for assoc in gathered_associativity:
                    associativity.update(assoc)

        # add to overall dictionary
        all_associativity[b] = associativity
        print('complete\n')

    # pickle overall dictionary
    # with open(f'pickles/complexity_measures/associativity_vs_depth_d{args.depth}_s1000.pickle', 'wb') as f:
    #     pickle.dump(all_associativity, f)


if __name__ == '__main__':
    main()
