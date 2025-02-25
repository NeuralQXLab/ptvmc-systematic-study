import functools
from functools import partial


def add_method(fun, cls=None, *, override=False):
    """Add a method to an existing class.

    Use as a decorator:

    .. code::

        @add_method(cls)
        def my_new_fun(self, args...):
            ...

    """
    if cls is None:
        return partial(add_method, cls=fun, override=override)

    if isinstance(fun, property):
        fname = fun.fget.__name__
    else:
        fname = fun.__name__

    if hasattr(cls, fname) and not override:
        raise AttributeError(
            f"Class {cls} already has a method named {fname} and {override =}"
        )

    setattr(cls, fname, fun)
    return fun


def replace_method(fun, cls=None):
    """Replace an existing method in a class.

    Use as a decorator:

    .. code::

        @replace_method(cls)
        def existing_method(self, args...):
            new code
            ...
    """
    if cls is None:
        return partial(replace_method, cls=fun)

    if isinstance(fun, property):
        fname = fun.fget.__name__
    else:
        fname = fun.__name__

    if not hasattr(cls, fname):
        raise AttributeError(
            f"Class {cls} does not have a method named {fname} to replace"
        )

    setattr(cls, fname, fun)
    return fun


def attach_method(fun, cls=None, *, prepend=True):
    """Appends some code to an existing method of a class.

    Use as a decorator:

    .. code::

        @attach_method(cls)
        def existing_fun(self, args...):
            ...

        #equivalent to
        class cls:
            def existing_fun(self, args...):
                previous_version()
                new_version()

    """
    if cls is None:
        return partial(attach_method, cls=fun)

    fname = fun.__name__

    if not hasattr(cls, fname):
        raise AttributeError(f"Class {cls} does not ahve a method named {fname}")

    bare_fun = getattr(cls, fname)

    if prepend:

        @functools.wraps(bare_fun)
        def _fun(*args, **kwargs):
            fun(*args, **kwargs)
            return bare_fun(*args, **kwargs)

    else:
        _fun = fun

    setattr(cls, fname, _fun)
    return fun


def attach_property(fun=None, *, cls=None, name: str, mode: str, prepend: bool = True):
    assert mode in ["set", "get", "del"]
    if cls is None:
        return partial(attach_property, cls=fun, name=name, mode=mode, prepend=prepend)

    if not hasattr(cls, name):
        raise AttributeError(f"Class {cls} does not have a method named {name}")

    old_prop = getattr(cls, name)
    fget = old_prop.fget
    fset = old_prop.fset
    fdel = old_prop.fdel
    doc = old_prop.__doc__

    args = [fget, fset, fdel, doc]
    if mode == "get":
        toedit = 0
    elif mode == "set":
        toedit = 1
    elif mode == "del":
        toedit = 2

    if prepend:
        bare_fun = args[toedit]

        @functools.wraps(bare_fun)
        def _fun(*args, **kwargs):
            fun(*args, **kwargs)
            return bare_fun(*args, **kwargs)

    else:
        _fun = fun

    args[toedit] = _fun
    new_property = property(*args)
    setattr(cls, name, new_property)
    return fun


def attach_setter(fun, cls=None, *, prepend=True):
    if cls is None:
        return partial(add_method, cls=fun)

    fname = fun.__name__

    if not hasattr(cls, fname):
        raise AttributeError(f"Class {cls} does not ahve a method named {fname}")

    bare_fun = getattr(cls, fname)

    if prepend:

        @functools.wraps(bare_fun)
        def _fun(*args, **kwargs):
            fun(*args, **kwargs)
            return bare_fun(*args, **kwargs)

    setattr(cls, fname, _fun)
    return _fun
