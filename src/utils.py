
def eval_decorator(func):

    def inner1(*args, **kwargs):
        returned_value = func(*args, **kwargs)

        return returned_value,

    return inner1
