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
    parser.add_argument('-s', '--sample', type=int, required=False, default=0,
                        help='number of triplets to sample (default: all)')
    args = parser.parse_args()

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
        for dc, table in all_tables[b].items():
            
            # iterate through depths
            assoc_vs_depth = []
            for depth in range(2, args.depth + 1):

                # generate tuples and triplets
                tuples = list(product(*[range(b)]*depth))
                if depth >= 3:
                    tuples = random.sample(tuples, b**3)
                triplets = list(combinations(tuples, 3))
                if len(triplets) > args.sample and args.sample != 0:
                    triplets = random.sample(triplets, args.sample)
                
                # check associativity for each triplet
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

        # add to overall dictionary
        all_associativity[b] = associativity
        print('complete\n')

    # pickle overall dictionary
    sample = f'_s{args.sample}' if args.sample != 0 else ''
    with open(f'pickles/complexity_measures/associativity_vs_depth_d{args.depth}{sample}.pickle', 'wb') as f:
        pickle.dump(all_associativity, f)


if __name__ == '__main__':
    main()
