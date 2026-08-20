import sys, pathlib, inspect, re
from collections import OrderedDict
from functools import wraps
import jax.numpy as jnp

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

local = pathlib.Path(__file__).parents[2]
assets = local / "assets"
examples = assets / "examples"

dist = local / "dist"
dist.mkdir(parents=True, exist_ok=True)

def jax_limit_cache(arg, *excluded, axis=0, maxsize=None):
    cache = OrderedDict()
    def decorator(f):
        sig = inspect.signature(f)
        @wraps(f)
        def wrapper(*a, **kw):
            bound = sig.bind(*a, **kw)
            bound.apply_defaults()
            limit = bound.arguments[arg]
            key = frozenset(
                    (k, v) for k, v in bound.arguments.items() if k != arg)
            res = None
            if key in cache:
                cache.move_to_end(key)
                size, res = cache[key]
                if size < limit:
                    res = None
                elif size == limit:
                    return res
            if res is None:
                res = f(*a, **kw)
                cache[key] = (limit, res)
                if maxsize is not None and len(cache) > maxsize:
                    cache.popitem(last=False)
                return res
            def mapping(path, x):
                if jax.tree_util.keystr(path) in excluded:
                    return x
                assert x.shape[axis] == size
                return jax.lax.slice_in_dim(x, 0, limit, axis=axis)
            return jax.tree.map_with_path(mapping, res)
        return wrapper
    return decorator

def lazy_default(**lazy):
    def decorator(f):
        sig = inspect.signature(f)
        @wraps(f)
        def wrapper(*a, **kw):
            bound = sig.bind_partial(*a, **kw)
            return f(*a, **kw, **{
                k: v(*a, **kw) for k, v in lazy.items()
                if k not in bound.arguments})
        return wrapper
    return decorator

def kron_bool(a, b):
    assert a.ndim == 2 and b.ndim == 2
    return jnp.reshape(
        jnp.logical_and(a[:, None, :, None], b[None, :, None, :]),
        (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))

_U16_DIVISORS_4X4 = lambda density: jnp.array(
    [[1 << (density * ((3 - c) * 4 + (3 - r))) for c in range(4)] for r in range(4)],
    dtype=jnp.uint16 if density == 1 else jnp.int32)

def unpack_u16_4x4(u16_grid, density=1):
    bits = (u16_grid[:, None, :, None] // _U16_DIVISORS_4X4(density)[None, :, None, :]) % 2
    return (bits != 0).reshape(u16_grid.shape[0] * 4, u16_grid.shape[1] * 4)

def shift_grid(u, dr, dc):
    h, w = u.shape
    if dr == -1:
        u = jnp.concatenate((u[1:, :], jnp.zeros((1, w), dtype=u.dtype)), axis=0)
    elif dr == 1:
        u = jnp.concatenate((jnp.zeros((1, w), dtype=u.dtype), u[:-1, :]), axis=0)
    if dc == -1:
        u = jnp.concatenate((u[:, 1:], jnp.zeros((h, 1), dtype=u.dtype)), axis=1)
    elif dc == 1:
        u = jnp.concatenate((jnp.zeros((h, 1), dtype=u.dtype), u[:, :-1]), axis=1)
    return u


patch_label = "patch_tag_runtime_callsite"
patch_sep = ".<locals>."

def patch_tag(name):
    assert bool(re.match(r"^[a-zA-Z_]\w*$", name))
    code = f"""
        def {patch_label}(f):
            @wraps(f)
            def {name}(*a, **kw):
                return f(*a, **kw)
            return {name}
    """
    exec(code.replace(code[:len(code) - len(code.lstrip())], "\n").rstrip())
    return locals()[patch_label]
