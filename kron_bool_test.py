import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from simplified import *
sys.path.pop(0)

def kron_bool_ref(a, b):
    # TODO: logical_and via broadcasting then reshape to result
    return jnp.bool(jnp.kron(jnp.uint8(a), jnp.uint8(b)))

key = jax.random.key(0)
a_key, b_key = jax.random.split(key)
a = jax.random.uniform(a_key, shape=(16, 16)) < 0.5
b = jax.random.uniform(b_key, shape=(4, 4)) < 0.5

print(a, b)
ref = kron_bool_ref(a, b)
out = kron_bool(a, b)
print(jnp.all(ref == out))
