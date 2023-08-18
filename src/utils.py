
def eval_decorator(func):
    global inner1
    def inner1(*args, **kwargs):
        returned_value = func(*args, **kwargs)

        return returned_value,

    inner1.__name__ = func.__self__.__class__.__name__
    return inner1
