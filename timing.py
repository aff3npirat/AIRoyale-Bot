import os
import timeit
import inspect
from argparse import ArgumentParser
from functools import wraps

import matplotlib.pyplot as plt



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
        logger.info(f"<EXEC> {name}: {toc - tic}")
        return result
    return wrapper


def intervall(func):
    name = func.__qualname__
    if name in _names:
        module = inspect.getmodule(func)
        name = f"{module.__name__}.{name}"
    _names.append(name)

    func.tic = None
    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.tic is None:
            func.tic = timeit.default_timer()
        else:
            logger.info(f"<INTERVALL> {name}: {timeit.default_timer() - func.tic}")
            func.tic = timeit.default_timer()
        return func(*args, **kwargs)
    return wrapper


def summarize_log(log_files, output):
    os.makedirs(output, exist_ok=True)

    lines = []
    for file in log_files:
        with open(file, "rt") as f:
            lines.extend(f.readlines()[1:])

    funcs = {}
    for line in lines:
        name, seconds = line.rstrip("\n").split(": ")

        if name not in funcs:
            funcs[name] = []

        funcs[name].append(float(seconds))

    for name, seconds in funcs.items():
        mean = sum(seconds)/len(seconds)
        plt.title(f"{name} | mean={mean:.4E}")
        plt.plot(seconds, ls="--", color="orange")
        plt.axhline(mean, ls="-", color="blue")
        plt.tight_layout()
        plt.savefig(os.path.join(output, f"{name}.png"))
        plt.close("all")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("files", type=str, nargs="+")
    parser.add_argument("--out", type=str, required=True)
    
    args = parser.parse_args()
    summarize_log(args.files, args.out)
