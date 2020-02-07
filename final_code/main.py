import numpy as np
from calculus import limit, derivative_at_point, derivative, partial_derivative_at_point, partial_derivative

def f(x: float) -> float:
    return x ** 2

def s(x: float) -> float:
    return np.sin(x) / x

def m(x: float, y: float, z: float) -> float:
    return x ** 2 + 3 * y - 2 * z ** 3

print(limit(s, 0))
print(limit(lambda x: 1 / x, np.inf))
print(derivative_at_point(f, 2))
g = derivative(f)
print(g(2))

print(partial_derivative_at_point(m, 1, 3, 2))
u = partial_derivative(m)[2]
print(u(3, 2, 4))