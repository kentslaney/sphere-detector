import sys, pathlib, inspect
from collections import OrderedDict
from functools import wraps

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

local = pathlib.Path(__file__).parents[2]
examples = local / "assets" / "examples"

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
