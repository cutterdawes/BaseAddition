import numpy as np
# import matplotlib.pyplot as plt
from itertools import product, combinations
import math
from tqdm.autonotebook import tqdm
from base import CarryTable, BaseElt


############################## cocycle-finding functions #############################

def assert_cocycle(table, depth=1):
    b = table.shape[0]
    depth += 1 # add last digit to check if carry is same
    tuples = list(product(*[range(b)]*depth))
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

def construct_tables(b, depth=1, rank=False, size=False):

    # initialize variables
    table_dict = {}
    cs = list(product(*[range(b)]*(b-1)))
    valid_dc = []
    if rank is not False:
        cs = np.array_split(cs, size)[rank]

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
                if assert_cocycle(table, depth=depth):
                    valid_dc.append(dc)

            pbar.update()

    pbar.close()

    # if depth > 1, filter table_dict by valid_dc
    if depth > 1:
        table_dict = {dc: table_dict[dc] for dc in valid_dc}
    
    return table_dict


####################### complexity measures of carry tables ########################

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


############################## displaying carry tables #############################

def show_tables(table_dict, b, depth=1):
    
    # create fig, axes
    n = len(table_dict)
    w = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(w, math.ceil(n / w), figsize=(2*w, 2*n//w))
    try:
        axes = axes.flatten()
    except:
        axes = [axes]
    
    # iterate through table_dict
    i = 0
    cs = list(product(*[range(b)]*(b-1)))
    for dc, table in table_dict.items():

        # find simplest c associated with dc
        for c in cs:
            c = tuple(c)
            c = (0,) + c
            if construct_coboundary(c) == dc:
                break

        # construct product table if specified
        if (depth > 1):
            table = construct_product_table(table, depth)

        # display image, increment i
        ax = axes[i]
        im = ax.imshow(table, cmap='viridis', vmin=0, vmax=b-1)
        ax.set_title(f'c = {str(c)}', fontsize=10)
        i += 1

    # turn off axes
    for ax in axes:
        ax.axis('off')
    
    # add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=80/w)
    cbar.set_ticks(range(b))