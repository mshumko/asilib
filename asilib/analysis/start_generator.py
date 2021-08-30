import functools


def start_generator(func):
    """
    A decorator to start a generator and advance it to
    the first occurrence of yield. Decorator code is
    almost for verbatim taken from David Beazley.

    Source: http://dabeaz.com/coroutines/coroutine.py
    """

    @functools.wraps(func)
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr

    return start
