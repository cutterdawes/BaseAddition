import argparse
import pickle
import numpy as np
import sys
sys.path.append('../')
import fn


def pickle_measures(measures, args):
    directory = '/home/cdawes/Repo/pickles' if (args.directory is None) else args.directory
    for filename, measure in measures.items():
        with open(f'{directory}/{filename}.pickle', 'wb') as file:
            pickle.dump(measure, file)
    print(f'Function executed successfully.\nOutputs saved to:')
    for filename in measures.keys():
        print(f'{directory}/{filename}.pickle')


def compute_measures(args):
    # load all_tables, get maximum depth if specified
    with open('../pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    bases = list(all_tables.keys())
    max_depth = args.depth if args.depth else 4

    # initialize dictionaries
    measures = {}
    frac_zeros_vs_depth = {b: {dc: [] for dc in all_tables[b].keys()} for b in bases}
    num_digits_vs_depth = {b: {dc: [] for dc in all_tables[b].keys()} for b in bases}
    est_dim_box_vs_depth = {b: {dc: [] for dc in all_tables[b].keys()} for b in bases}

    # iterate through bases and c's
    for b in bases:
        if b > 5:
            break
        for dc in all_tables[b].keys():
            for depth in range(1, max_depth+1):
                table = fn.construct_product_table(all_tables[b][dc], depth=depth)
                frac_zeros = (table.size - np.count_nonzero(table)) / table.size
                frac_zeros_vs_depth[b][dc].append(frac_zeros)
                num_digits = len(np.unique(table))
                num_digits_vs_depth[b][dc].append(num_digits)
                est_dim = fn.get_min_dim(table, b)
                est_dim_box_vs_depth[b][dc].append(est_dim)

    # add to measures
    measures['frac_zeros_vs_depth'] = frac_zeros_vs_depth
    measures['num_digits_vs_depth'] = num_digits_vs_depth
    measures['est_dim_box_vs_depth'] = est_dim_box_vs_depth

    return measures


def main(): 
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute complexity measures on carry tables up to specified depth')
    parser.add_argument('--depth', type=int, required=False, help='maximum depth of computed measures')
    parser.add_argument('--directory', type=str, required=False, help='directory of pickled measures')
    args = parser.parse_args()
        
    # compute and pickle measures
    measures = compute_measures(args)
    pickle_measures(measures, args)


if __name__ == '__main__':
    main()