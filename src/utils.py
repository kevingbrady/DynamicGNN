
def is_number(x):

    try:
        float(x)

    except ValueError:
        return False

    return True
