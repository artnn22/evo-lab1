DIM = 8
BOUNDS = (-10.0, 10.0)


def dixon_price(x):
    value = (x[0] - 1) ** 2
    for i in range(1, len(x)):
        value += (i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2
    return value


def analytical_minimum():
    point = []
    for i in range(1, DIM + 1):
        exponent = -(2 ** i - 2) / (2 ** i)
        point.append(2 ** exponent)
    return point


def clip_to_bounds(point, bounds):
    result = []
    for value in point:
        if value < bounds[0]:
            value = bounds[0]
        elif value > bounds[1]:
            value = bounds[1]
        result.append(value)
    return result
