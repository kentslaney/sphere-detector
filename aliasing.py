import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

a = jnp.array([1.8, 2.3])
import matplotlib.pyplot as plt
plt.scatter(*a)
# print(b.origin, b.steps())
b = [AliasedRay(a, i / 10).steps() for i in range(62)]
plt.scatter(*zip(*b))
plt.show()
