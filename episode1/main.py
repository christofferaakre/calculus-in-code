import numpy as np

delta = 0.0001

def f(x: float) -> float:
    return np.sin(x) / x

def g(x: float) -> float:
    return x ** 2

def h(x: float, y: float, z: float) -> float:
    return x ** 2 + y - 3 * z

def limit(f: callable, x: float) -> float:
    left_limit = f(x - delta)
    right_limit = f(x + delta)
    return [
        left_limit, 
        right_limit
    ]

def derivative_at_point(f: callable, x: float) -> float:
    return (f(x + delta) - f(x)) / delta

def derivative(f: callable) -> callable:
    return lambda x: (f(x + delta) - f(x)) / delta

def partial_derivative_at_point(f: callable, x: float, y: float, z: float) -> list:
    return [
        (f(x + delta, y, z) - f(x, y, z)) / delta,
        (f(x, y + delta, z) - f(x, y, z)) / delta,
        (f(x, y, z + delta) - f(x, y, z)) / delta
    ]

def partial_derivative(f: callable) -> list:
    return [
        lambda x, y, z: (f(x + delta, y, z) - f(x, y, z)) / delta,
        lambda x, y, z: (f(x, y + delta, z) - f(x, y, z)) / delta,
        lambda x, y, z: (f(x, y, z + delta) - f(x, y, z)) / delta
    ]

print(limit(f, 0))
print(derivative_at_point(g, 3))

g_prime = derivative(g)
print(g_prime(3))

print(partial_derivative_at_point(h, 1, 1, 1))
h_x = partial_derivative(h)[0]
h_y = partial_derivative(h)[1]
h_z = partial_derivative(h)[2]
print(h_x(1, 1, 1))
print(h_y(1, 1, 1))
print(h_z(1, 1, 1))