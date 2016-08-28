import lmfit
import cPickle
import random
from copy import deepcopy


# invent an arbitrary Parameters set with expressions
p = lmfit.Parameters()
for i in range(10):
    p.add('var_%s' % i, value=i)
    if i >= 1:
        j = random.randint(0, i)
        p['var_%s' % i].expr = 'var_%s' % j

# pull out expressions and store in list of tuples
t = []
for v in p:
    t.append((v, p[v].expr))

# remove expressions from Parameters instance
original = deepcopy(p)
for v in p:
    p[v].expr = None

# we'll need to bundle these somehow so that they're both pickled
class bundle(object):
    def __init__(self, par, expr):
        """
            -par: the parameters instance
            -expr: the sorted list of constraint expressions
        """
        self.par = par
        self.expr = expr

b = bundle(p, t)

# pickle
with open('test_pickle.dat', 'wb') as f:
    cPickle.dump(b, f, -1)

# unpickling
with open('test_pickle.dat', 'rb') as f:
    bprime = cPickle.load(f)

# recreate our original Parameters instance
pprime = bprime.par
for v in bprime.expr:
    pprime[v[0]].expr = v[1]

print original

print pprime
