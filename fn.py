import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
import random
import math
from tqdm.autonotebook import tqdm


##################### base representation classes and functions #############################

class RecursiveTable():
    def __init__(self, carry_table):
        self.b = len(carry_table)
        self.carry_table = carry_table
        # prev_level should be either a table (i.e. array) or RecursiveTable

    def __getitem__(self, elts):
        g1, g2 = elts
        if len(g1) != len(g2):
            zp = max(len(g1), len(g2))
            v1 = tuple([0] * (zp - len(g1)) + list(g1))
            v2 = tuple([0] * (zp - len(g2)) + list(g2))
            return self[(v1, v2)]
        if len(g1) == 1:
            return self.carry_table[g1[0], g2[0]]
        a, b = g1[0], g2[0]
        t1, t2 = g1[1:], g2[1:]
        z = self[(t1, t2)]
        res = (self[((a + b) % self.b,), (z,)] + self[(a,), (b,)]) % self.b
        return res

# elements of the form (d1,...,dk) in which addition is performed by recursively applying the carry table
class RecursiveGrpElt():
    def __init__(self, vals, carry_table):
        self.carry_table = carry_table
        self.vals = vals
        self.rt = RecursiveTable(carry_table)
        self.b = len(carry_table)

    def __add__(self, other):
        if len(self.vals) != len(other.vals):
            # zero pad if necessary
            zp = max(len(self.vals), len(other.vals))
            v1 = [0] * (zp - len(self.vals)) + list(self.vals)
            v2 = [0] * (zp - len(other.vals)) + list(other.vals)
            g1 = RecursiveGrpElt(tuple(v1), self.carry_table)
            g2 = RecursiveGrpElt(tuple(v2), self.carry_table)
            return g1 + g2
        else:
            if len(self.vals) == 1:
                carried = self.carry_table[self.vals[0], other.vals[0]]
                new_vals = [carried] + [(self.vals[0] + other.vals[0]) % self.b]
                if carried == 0:
                    new_vals = new_vals[1:]
                return RecursiveGrpElt(tuple(new_vals), self.carry_table)

            # carried element from the tail
            z = self.rt[(self.vals[1:], other.vals[1:])]
            # overall carried element
            carried = self.carry_table[self.vals[0], other.vals[0]]
            carried += self.carry_table[(self.vals[0] + other.vals[0]) % self.b, z]
            carried = (carried) % self.b
            new_a = (self.vals[0] + other.vals[0] + z) % self.b
            g1 = RecursiveGrpElt(self.vals[1:], self.carry_table)
            g2 = RecursiveGrpElt(other.vals[1:], self.carry_table)
            new_tail = list((g1 + g2).vals)[-len(self.vals[1:]):]
            new_vals = tuple([carried] + [new_a] + new_tail)
            if carried == 0:
                new_vals = new_vals[1:]
            return RecursiveGrpElt(new_vals, self.carry_table)
        
#construct table for all tuples of a given length
def construct_product_table(table, depth):
    b = len(table)
    tab = np.zeros((b**depth, b**depth))
    rt = RecursiveTable(table)
    for i, v1 in enumerate(product(*[range(b)]*depth)):
        for j, v2 in enumerate(product(*[range(b)]*depth)):
            tab[i, j] = rt[(v1, v2)]
    return tab


############################## cocycle-finding functions #############################

def assert_cocycle(table, depth=1):
    b=table.shape[0]
    depth += 1 # add last digit to check if carry is same
    tuples = list(product(*[range(b)]*depth))
    for (v1, v2, v3) in combinations(tuples, 3): #iterate over all tuples of given depth

        # convert to group elements
        g1 = RecursiveGrpElt(v1, table)
        g2 = RecursiveGrpElt(v2, table)
        g3 = RecursiveGrpElt(v3, table)

        # check associativity
        s1 = (g1 + g2) + g3
        s2 = g1 + (g2 + g3)
        is_assoc = s1.vals[-depth:] == s2.vals[-depth:]
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

def construct_tables(b, depth=1, rank=False, size=False):

    # initialize variables
    table_dict = {}
    cs = list(product(*[range(b)]*(b-1)))
    if rank is not False:
        cs = np.array_split(cs, size)[rank]

    # iterate through c's
    pbar = tqdm(total=b**(b-2), desc='Add cocycles')
    for c in cs:

        # construct c and coboundary dc
        c = tuple(c)
        c = (0,) + c
        dc = construct_coboundary(c)

        # add associated table to table_dict if not added yet
        if dc not in table_dict.keys():
            table_dict[dc] = construct_table(dc)
            pbar.update()

    pbar.close()    

    # if depth > 1, check that each table is a recursive cocycle up to depth
    if depth > 1:
        valid_dc = []
        for dc, table in tqdm(table_dict.items(), desc='Check recursive'):
            if assert_cocycle(table, depth=depth):
                valid_dc.append(dc)
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