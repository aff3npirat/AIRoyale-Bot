import timeit
import inspect
from functools import wraps



logger = None

_names = []


def exec_time(func):
    name = func.__qualname__
    if name in _names:
        module = inspect.getmodule(func)
        name = f"{module.__name__}.{name}"
    _names.append(name)

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = func(*args, **kwargs)
        toc = timeit.default_timer()
        logger.info(f"{name}: {toc - tic}")
        return result
    return wrapper