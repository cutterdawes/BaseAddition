import numpy as np

# Simon's version of the recursive rule for carrying
def recursive_carry_rule(carry_table, n, m):
    carried_tail = carry_table[n[1:], m[1:]]
    return (carry_table[n[0], m[0]] + carry_table[(n[0] + m[0]) % carry_table.b, carried_tail]) % carry_table.b

class CarryTable():
    def __init__(self, carry_table):
        self.carry_table = carry_table
        self.b = len(carry_table)
        # prev_level should be either a table (i.e. array) or CarryTable

    def __len__(self):
        return len(self.carry_table)

    def __getitem__(self, elts: tuple):

        # get tuples
        n, m = elts

        # convert to tuple if necessary
        if type(n) in (int, np.int64):
            n = (n,)
        if type(m) in (int, np.int64):
            m = (m,)

        # zero pad if necessary
        if len(n) != len(m):
            zp = max(len(n), len(m))
            n = (0,) * (zp - len(n)) + n
            m = (0,) * (zp - len(m)) + m

        # 1-digit case
        if len(n) == 1:
            carried = self.carry_table[n[0], m[0]]
        
        # general case
        else:
            carried = recursive_carry_rule(self, n, m)

        return carried
    
# elements of the form (nk, ..., n1) in which addition is performed by recursively applying the carry table
class BaseElt():
    def __init__(self, vals, carry_table):
        self.carry_table = CarryTable(carry_table)
        self.b = len(carry_table)
        vals = (vals,) if type(vals) in (int, np.int64) else vals
        self.vals = vals

    def __len__(self):
        return len(self.vals)
    
    def __getitem__(self, idx):
        return self.vals[idx]

    def __add__(self, other):

        # get tuples
        n = self.vals
        m = other.vals

        # zero pad if necessary
        if len(n) != len(m):
            zp = max(len(n), len(m))
            n = (0,) * (zp - len(n)) + n
            m = (0,) * (zp - len(m)) + m

        # 1-digit case
        if len(n) == 1:
            carried = self.carry_table[n[0], m[0]]
            s = (n[0] + m[0]) % self.b
            if carried != 0:
                s = (carried,) + (s,)
        
        # general case
        else:
            # get values and carried of tail
            n_tail = BaseElt(n[1:], self.carry_table)
            m_tail = BaseElt(m[1:], self.carry_table)
            s_tail = (n_tail + m_tail).vals
            s_tail = s_tail[-len(n_tail):] if len(s_tail) > 1 else s_tail
            carried_tail = self.carry_table[n[1:], m[1:]]

            # get value and carried of head
            s_head = (n[0] + m[0] + carried_tail) % self.b
            carried = recursive_carry_rule(self.carry_table, n, m)

            # add head and tail and carried (if applicable)
            s = (s_head,) + s_tail
            if carried != 0:
                s = (carried,) + s

        return BaseElt(s, self.carry_table)