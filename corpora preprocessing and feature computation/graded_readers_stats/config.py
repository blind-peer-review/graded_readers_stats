is_debug = False


def log(*args):
    if is_debug:
        print(*args)
