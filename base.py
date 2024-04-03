import numpy as np

class CarryTable():
    def __init__(self, carry_table):
        self.carry_table = carry_table
        self.b = len(carry_table)
        # prev_level should be either a table (i.e. array) or CarryTable

    def __getitem__(self, elts):

        # rename variables for readability
        n, m = elts
        carry_table = self.carry_table
        b = self.b

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
            return carry_table[n[0], m[0]]
        
        # general case
        carried_tail = self[n[1:], m[1:]]
        carried = (self[((n[0] + m[0]) % b,), (carried_tail,)] + carry_table[n[0], m[0]]) % b  # Simon's recursive carry
        return carried
    
    def __len__(self):
        return len(self.carry_table)

# elements of the form (nk, ..., n1) in which addition is performed by recursively applying the carry table
class BaseRep():
    def __init__(self, vals, carry_table):
        self.vals = vals
        self.carry_table = CarryTable(carry_table)
        self.b = len(carry_table)

    def __add__(self, other):

        # rename variables for readability
        n = self.vals
        m = self.vals
        carry_table = self.carry_table
        b = self.b

        # zero pad if necessary
        if len(n) != len(m):
            zp = max(len(n), len(m))
            n = (0,) * (zp - len(n)) + n
            m = (0,) * (zp - len(m)) + m
            n = BaseRep(n, carry_table)
            m = BaseRep(m, carry_table)

        # 1-digit case
        if len(n) == 1:
            carried = carry_table[n[0], m[0]]
            if carried != 0:
                s = (carried,) + ((n[0] + m[0]) % b,)
            return BaseRep(s, carry_table)

        # general case (Simon's implementation)
        carried_tail = carry_table[n[1:], m[1:]]
        carried = (carry_table[n[0], m[0]] + carry_table[(n[0] + m[0]) % b, carried_tail]) % b
        s = (n[0] + m[0] + carried_tail) % b
        n = BaseRep(n[1:], carry_table)
        m = BaseRep(m[1:], carry_table)
        s_tail = (n + m).vals[-len(n.vals[1:]):]
        s = (s,) + s_tail
        if carried != 0:
            s = (carried,) + s
        return BaseRep(s, carry_table)
