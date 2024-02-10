import numpy as np,matplotlib.pyplot as plt,seaborn as sns
from itertools import product
import random
import math
from tqdm.notebook import tqdm_notebook


##################### base representation classes and functions #############################

class RecursiveTable():
    def __init__(self, carry_table):
        self.d = len(carry_table)
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
        res = (self[((a + b) % self.d,), (z,)] + self[(a,), (b,)]) % self.d
        return res

# elements of the form (d1,...,dk) in which addition is performed by recursively applying the carry table
class RecursiveGrpElt():
    def __init__(self, vals, carry_table):
        self.carry_table = carry_table
        self.vals = vals
        self.rt = RecursiveTable(carry_table)
        self.d = len(carry_table)

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
                new_vals = [carried] + [(self.vals[0] + other.vals[0]) % self.d]
                if carried == 0:
                    new_vals = new_vals[1:]
                return RecursiveGrpElt(tuple(new_vals), self.carry_table)

            # carried element from the tail
            z = self.rt[(self.vals[1:], other.vals[1:])]
            # overall carried element
            carried = self.carry_table[self.vals[0], other.vals[0]]
            carried += self.carry_table[(self.vals[0] + other.vals[0]) % self.d, z]
            carried = (carried) % self.d
            new_a = (self.vals[0] + other.vals[0] + z) % self.d
            g1 = RecursiveGrpElt(self.vals[1:], self.carry_table)
            g2 = RecursiveGrpElt(other.vals[1:], self.carry_table)
            new_tail = list((g1 + g2).vals)[-len(self.vals[1:]):]
            new_vals = tuple([carried] + [new_a] + new_tail)
            if carried == 0:
                new_vals = new_vals[1:]
            return RecursiveGrpElt(new_vals, self.carry_table)
        
#construct table for all tuples of a given length
def construct_product_table(table, depth):
    d=len(table)
    tab=np.zeros((d**depth,d**depth))
    rt=RecursiveTable(table)
    for i,v1 in enumerate(product(*[range(d)]*depth)):
        for j,v2 in enumerate(product(*[range(d)]*depth)):
            tab[i,j]=rt[(v1,v2)]
    return tab


############################## cocycle-finding functions #############################

def assert_cocycle(table, depth=2, sample=False, frac=False, n=False):
    d=table.shape[0]
    tuples = list(product(*[range(d)]*depth))
    if sample:
        if frac:
            tuples = random.sample(tuples, math.floor(d**depth * frac))
        if n:
            assert n <= d**depth, "n too large"
            tuples = random.sample(tuples, n)
    for v1 in tuples: #iterate over all tuples of given depth
        for v2 in tuples:
            for v3 in tuples:
                g1=RecursiveGrpElt(v1, table)
                g2=RecursiveGrpElt(v2, table)
                g3=RecursiveGrpElt(v3, table)

                s1=(g1+g2)+g3
                s2=g1+(g2+g3)
                is_assoc=s1.vals==s2.vals
                if not is_assoc:
                    return False
    return True

def construct_table(d, h):
    basic_table=1*(np.add.outer(np.arange(d),np.arange(d))>=d)
    table = np.zeros((d, d), dtype='int')
    for i in range(d):
        for j in range(d):
            table[i, j] = (basic_table[i, j] + h[(i+j)%d] - h[i] - h[j]) % d
    return table

def construct_tables(d, sample=False, frac=False, n=False):
    table_dict = {}
    hs = list(product(*[range(d)]*(d-1)))
    for h in tqdm_notebook(hs):
        h = (0,) + h
        table = construct_table(d, h)
        if assert_cocycle(table, sample=sample, frac=frac, n=n):
            if not any([np.array_equal(table, o_table) for o_table in table_dict.values()]):
                table_dict[str(h)] = table
    if sample:
        valid_h = []
        for h in table_dict.keys():
            if assert_cocycle(table_dict[h]):
                valid_h.append(h)
        table_dict = {h: table_dict[h] for h in valid_h}
    return table_dict


############################## displaying carry tables #############################

def show_tables(table_dict, d, depth=False):
    
    # create fig, axes
    n = len(table_dict)
    w = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(math.ceil(n / w), w, figsize=(2*w, 2*n//w))
    axes = axes.flatten()
    
    # iterate through table_dict
    i = 0
    for h in table_dict.keys():

        # get h and table, construct product table if specified
        h = list(table_dict.keys())[i]
        table = table_dict[h]
        if depth:
            table = construct_product_table(table, depth)

        # display image, increment i
        ax = axes[i]
        im = ax.imshow(table, cmap='viridis', vmin=0, vmax=d-1)
        ax.set_title('h = ' + h, fontsize=10)
        i += 1

    # turn off axes
    for ax in axes:
        ax.axis('off')
    
    # add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(range(d))