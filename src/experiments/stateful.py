import functools, inspect
from collections import namedtuple
from types import MethodType

class Context:
    scope = None

    def __init__(self, name, state):
        # TODO: jax.extend.core.Primitive; mb.read_state; mb.coreml_update_state
        assert state is not Ellipsis, "externally managed storage unimplemented"
        self.closure = {} if state is None else state
        self.name = name

    def __enter__(self):
        assert __class__.scope is None, "nesting unimplemented" # TODO
        self.locked = set()
        __class__.scope = self
        return self

    def __getitem__(self, key):
        return self.closure[key] if self.starting else \
                getattr(self.closure, key)

    def __setitem__(self, key, value):
        if self.starting:
            self.closure[key] = value
        else:
            self.closure = self.closure._replace(**{key: value})

    @functools.cached_property
    def starting(self):
        return isinstance(self.closure, dict)

    @property
    def serializable(self):
        if self.starting:
            return namedtuple(self.name, self.closure.keys())(**self.closure)
        return self.closure

    def __exit__(self, exc_type, exc_val, exc_tb):
        __class__.scope = None

def restores(**contained):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            assert Context.scope is not None, "missing boundary for implicits"
            namespace = f.__globals__
            for k in contained:
                assert k not in namespace
                if k in Context.scope.locked:
                    raise RuntimeError(
                        f"The state '{k}' was already restored in this trace. "
                        "Multiple restores cause branching side-effects in JAX."
                    )
                Context.scope.locked.add(k)
                namespace[k] = contained[k] if Context.scope.starting else \
                        Context.scope[k]
            result = f(*a, **kw)
            for k in contained:
                Context.scope[k] = namespace[k]
                del namespace[k]
            return result
        return wrapper
    return decorator

class Decorator:
    def __init__(self, f, argname):
        self.f, self.argname = f, argname
        functools.update_wrapper(self, f)
        self.prebound = \
                f.__qualname__.rsplit('.', 2)[-2:-1] not in ([], ["<locals>"])

    @functools.partial(property, None)
    def prebound(self, value):
        sig = inspect.signature(self.f)
        params = list(sig.parameters.values())
        index = int(value)
        kind = inspect.Parameter.POSITIONAL_ONLY if (
            index < len(params) and
            params[index].kind == inspect.Parameter.POSITIONAL_ONLY
        ) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        params.insert(index, inspect.Parameter(self.argname, kind))
        self.__signature__ = sig.replace(parameters=params)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(*args, **kwargs):
        self, args = args[0], args[1:]
        sig = self.__signature__
        start = next(iter(sig.parameters.values())).kind
        bound = sig.bind(*args, **kwargs)
        if start == inspect.Parameter.POSITIONAL_ONLY:
            index = list(sig.parameters).index(self.argname)
            state = bound.args[index]
            args = bound.args[:index] + bound.args[index + 1:]
        else:
            state, args = bound.arguments.pop(self.argname), bound.args
        with Context(self.argname, state) as scope:
            result = self.f(*args, **bound.kwargs)
            return (scope.serializable, result)

def implicit(argname):
    def decorator(f):
        return Decorator(f, argname)
    return decorator
