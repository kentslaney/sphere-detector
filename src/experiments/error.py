from .stateful import implicit, restores

from dataclasses import dataclass

import jax
import jax.numpy as jnp

@restores(momentum=jnp.ones(()))
def block(x):
    global momentum
    momentum += x
    return x + momentum

@jax.jit
def wrapper(x):
    return block(x)

@jax.jit(donate_argnames=["state"])
@implicit(argname="state")
def exported(x):
    return wrapper(x)

output = [None] * 2

state, output[0] = exported(None, 2)
state, output[1] = exported(state, 3)

print(jnp.asarray(output))
