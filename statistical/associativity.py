# import packages
import argparse
from itertools import product, combinations
import random
import pickle
import sys
sys.path.append('../')
from base import BaseElt


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, required=True)
    args = parser.parse_args()

    # initialize overall dictionary, all tables
    all_associativity = {}
    with open('../pickles/carry_tables/all_tables_d1.pickle', 'rb') as f:
        all_tables = pickle.load(f)

    # iterate through bases
    for b in range(3, 6):
        print(f'base {b}...')

        # initialize dictionary
        associativity = {}

        # iterate through tables
        for dc, table in all_tables[b].items():
            tuples = list(product(*[range(b)]*args.depth))
            if args.depth > 3:
                tuples = random.sample(tuples, b**3)
            num_assoc = 0
            triplets = list(combinations(tuples, 3))
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
        
            # add fraction of associativity to dictionary
            associativity[dc] = num_assoc / len(triplets)

        # add to overall dictionary
        all_associativity[b] = associativity
        print('complete\n')

        # pickle overall dictionary
        with open(f'../pickles/complexity_measures/associativity_d{args.depth}.pickle', 'wb') as f:
            pickle.dump(all_associativity, f)


if __name__ == '__main__':
    main()