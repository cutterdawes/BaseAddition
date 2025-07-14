import math
import random
from itertools import product, combinations
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from tqdm.autonotebook import tqdm

from base_rep import CarryTable, BaseElt


### cocycle-finding functions ######################################################

def assert_cocycle(table, depth=1, sample=False):
    b = table.shape[0]
    depth += 1 # add last digit to check if carry is same
    tuples = list(product(*[range(b)]*depth))
    if sample:
        assert (sample >= 3) and (sample <= b**depth), 'need 3 <= sample <= b**depth'
        tuples = random.sample(tuples, sample)
    for (n, m, p) in combinations(tuples, 3): #iterate over all tuples of given depth

        # convert to group elements
        n = BaseElt(n, table)
        m = BaseElt(m, table)
        p = BaseElt(p, table)

        # check associativity
        s1 = (n + m) + p
        s2 = n + (m + p)
        is_assoc = s1.vals == s2.vals
        if not is_assoc:
            return False
    return True

def construct_coboundary(c):
    b = len(c)
    dc = np.zeros((b, b), dtype='int')
    for i in range(b):
        for j in range(b):
            dc[i, j] = (c[i] + c[j] - c[(i+j)%b]) % b
    dc = tuple(map(tuple, dc))
    return dc

def construct_table(dc):
    b = len(dc)
    standard_table = 1 * (np.add.outer(np.arange(b), np.arange(b)) >= b)
    table = np.zeros((b, b), dtype='int')
    for i in range(b):
        for j in range(b):
            table[i, j] = (standard_table[i, j] + dc[i][j]) % b
    return table

def construct_product_table(table, depth):
    b = len(table)
    product_table = np.zeros((b**depth, b**depth))
    table = CarryTable(table)
    for i, n in enumerate(product(*[range(b)]*depth)):
        for j, m in enumerate(product(*[range(b)]*depth)):
            product_table[i, j] = table[n, m]
    return product_table

def construct_tables(b, depth=1, sample=False):

    # initialize variables
    table_dict = {}
    cs = list(product(*[range(b)]*(b-1)))
    valid_dc = []

    # iterate through c's
    pbar = tqdm(total=b**(b-2))
    for c in cs:

        # construct c and coboundary dc
        c = tuple(c)
        c = (0,) + c
        dc = construct_coboundary(c)

        # add associated table to table_dict if not added yet
        if dc not in table_dict.keys():
            table = construct_table(dc)
            table_dict[dc] = table

            # if depth > 1, check if table is a recursive cocycle up to depth
            if depth > 1:
                if assert_cocycle(table, depth=depth, sample=sample):
                    valid_dc.append(dc)

            pbar.update()

    pbar.close()

    # if depth > 1, filter table_dict by valid_dc
    if depth > 1:
        table_dict = {dc: table_dict[dc] for dc in valid_dc}
    
    return table_dict


### complexity measures of carry tables ###############################

def get_border(table):
    diff_ax0 = np.diff(table, axis=0, prepend=0)
    diff_ax1 = np.diff(table, axis=1, prepend=0)
    border = np.where(diff_ax0 + diff_ax1 != 0, 1, 0)
    return border

def plot_border(table):
    border = get_border(table)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im = axes[0].imshow(table, cmap='viridis', vmin=0, vmax=len(table)-1)
    axes[1].imshow(border, cmap='viridis', vmin=0, vmax=len(table)-1)
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].set_title('table')
    axes[1].set_title('border')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(range(len(table)))

def get_dim(table):
    border = get_border(table)
    N = np.count_nonzero(border)
    eps_inv = len(border)
    dim = np.log(N) / np.log(eps_inv)
    return dim

def get_order(u, b, k):
    order = tuple((j*u)%b for j in range(b))
    order = list(product(*[order]*k))
    order = [np.sum(np.array(n)*b**np.flip(range(k))) for n in order]
    return order

def get_min_dim(table, b):
    k = int(math.log(len(table), b))
    min_dim = np.inf
    for u in range(b):
        if math.gcd(u, b) == 1:
            order = get_order(u, b, k)
            dim = get_dim(table[order][:,order])
            if dim < min_dim:
                min_dim = dim
    return min_dim

### displaying carry tables ###############################################

def add_border(ax, color='blue', width=2):
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(width)

def show_tables(table_dict, b, depth=1, savefig=False):
    
    # create fig, axes
    n = len(table_dict)
    w = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(w, math.ceil(n / w), figsize=(2*w, 2*n//w))
    fig.suptitle('Carry Tables, '+ r'$b =$' + str(b), fontsize=16, y=0.94)
    try:
        axes = axes.flatten()
    except:
        axes = [axes]

    # sort table_dict and load est. dims
    table_dict = {dc: table_dict[dc] for dc in sorted(table_dict.keys())}
    with open('../pickles/complexity_measures/est_dim_box_vs_depth.pickle', 'rb') as f:
        est_dim_box_vs_depth = pickle.load(f)
    
    # iterate through table_dict
    i = 0
    for dc, table in table_dict.items():

        # construct product table if specified
        if (depth > 1):
            table = construct_product_table(table, depth)

        # classify as Single Value, Low Dim. Multiple Value, or Other Multiple Value
        est_dim = est_dim_box_vs_depth[b][dc][3]
        if len(np.unique(table)) == 2:
            color = 'blue'
        elif est_dim > 1.25 and est_dim < 1.5:
            color = 'orange'
        else:
            color = 'silver'

        # display image, increment i
        ax = axes[i]
        add_border(ax, color=color, width=5)
        levels = np.linspace(-0.5, b-0.5, b+1)
        norm = BoundaryNorm(levels, ncolors=256)
        im = ax.imshow(table, cmap='viridis', norm=norm)
        i += 1

    # turn off axis ticks and labels
    i = 0
    for ax in axes:
        # ax.tick_params(axis='both', which='both', length=0)
        if i % w == 0:
            ax.set_yticks(range(b))
            ax.tick_params(axis='y', length=0)
        else:
            ax.set_yticks([])
        if i // w == w - 1:
            ax.set_xticks(range(b))
            ax.tick_params(axis='x', length=0)
        else:
            ax.set_xticks([])
        i += 1
    
    # add colorbar
    plt.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.88)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.04, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax, boundaries=levels, drawedges=True)
    cbar.set_ticks(range(b))
    cbar.set_label('Carry Value', fontsize=12, rotation=270, labelpad=15)

    if savefig:
        plt.savefig(f'../figures/tables{b}_d{depth}.png', dpi=300, bbox_inches='tight')