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
        logger.info(f"{name}: {toc - tic}")
        return result
    return wrapper


def summarize_log(log_file, output):
    os.makedirs(output, exist_ok=True)

    with open(log_file, "rt") as f:
        lines = f.readlines()[1:]

    funcs = {}
    for line in lines:
        name, seconds = line.rstrip("\n").split(": ")

        if name not in funcs:
            funcs[name] = []

        funcs[name].append(float(seconds))

    for name, seconds in funcs.items():
        mean = sum(seconds)/len(seconds)
        plt.title(f"{name} | mean={mean:.4E}")
        plt.plot(seconds, "--", color="orange")
        plt.axhline(mean, "-", color="blue")
        plt.tight_layout()
        plt.savefig(os.path.join(output, f"{name}.png"))
        plt.close("all")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in-file", type=str)
    parser.add_argument("output", type=str)
    
    args = parser.parse_args()
    summarize_log(args.in_file, args.output)
