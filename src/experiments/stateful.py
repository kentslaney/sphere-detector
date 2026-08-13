import jax, jax.extend.core
import functools, inspect
from collections import namedtuple
from types import MethodType

jax.tree_util.register_static(type(Ellipsis))

class Context:
    scope = None
    external = None

    def __init__(self, name, state):
        assert name != "ffi"
        if state is Ellipsis and __class__.scope is None:
            state, self.external = None, {}
        if type(state).__name__ == "ffi":
            self.external = ...

        self.closure = {} if state is None else state
        self.name = name

    def read(self, key, shape):
        from jax._src.interpreters import mlir

        read = jax.extend.core.Primitive("stateful_intake")
        read.multiple_results = False
        read.def_abstract_eval(lambda _: shape)

        @functools.partial(mlir.register_lowering, read)
        def _lowering(ctx, ref):
            ir_types = mlir.aval_to_ir_types(ctx.module_context, shape)
            return mlir.custom_call(
                "ffi_read",
                operands=[ref],
                result_types=ir_types,
                backend_config=key,
            ).results

        return read

    def __enter__(self):
        if __class__.scope is not None:
            assert self.closure is Ellipsis # TODO
        else:
            self.locked = set()
            __class__.scope = self
            return self

    def __getitem__(self, key):
        res = self.closure[key] if self.starting else getattr(self.closure, key)
        if self.external is not Ellipsis:
            return res
        return self.read(key, jax.typeof(res)).bind(res)

    def __setitem__(self, key, value):
        if self.starting:
            self.closure[key] = value
        else:
            self.closure = self.closure._replace(**{key: value})

    def register(self, key, default):
        assert self.external is not Ellipsis
        if self.external is not None:
            self.external[key] = default
        return default

    @functools.cached_property
    def starting(self):
        return isinstance(self.closure, dict)

    @property
    def serializable(self):
        if self.starting:
            return namedtuple(
                    self.name if self.external is None else "ffi",
                    self.closure.keys())(**self.closure)
        return self.closure

    def _asdict(self):
        if self.starting:
            return self.closure
        return self.closure._asdict()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closure is not Ellipsis:
            trace = jax.extend.core.find_top_trace(())
            for k, v in self._asdict().items():
                if isinstance(v, jax.core.Tracer):
                    if v._trace != trace:
                        src = v._trace.frame.debug_info.func_src_info
                        raise NameError(
                            f"implicit '{k}' type {v} was produced by a trace "
                            f"missing '{self.name}' for {src}"
                        )
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
                namespace[k] = Context.scope.register(k, contained[k]) \
                        if Context.scope.starting else Context.scope[k]
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
        ) else inspect.Parameter.KEYWORD_ONLY if (
            index > 0 and params[0].kind == inspect.Parameter.KEYWORD_ONLY
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
            return result if scope is None else (scope.serializable, result)

def implicit(argname):
    def decorator(f):
        return Decorator(f, argname)
    if not isinstance(argname, str):
        return Decorator(argname, "state")
    return decorator

def managed(arg):
    raise NotImplementedError()
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            return f(*a, **kw)
        return wrapper
    if not isinstance(arg, str):
        f, arg = arg, None
        return decorator(f)
    return decorator
