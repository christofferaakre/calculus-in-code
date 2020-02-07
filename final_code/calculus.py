from inspect import signature
import numpy as np

delta = 0.0001

def limit(f: callable, x: float) -> list:
    left_limit = f(x - delta)
    right_limit = f(x + delta)
    return np.array([
        left_limit,
        right_limit
    ])

def derivative_at_point(f: callable, x: float) -> float:
    return (f(x + delta) - f(x)) / delta

def derivative(f: callable) -> callable:
    return lambda x: (f(x + delta) - f(x)) / delta

def partial_derivative_at_point(f: callable, x: float, y: float, z: float) -> list:
    return np.array([
        (f(x + delta, y, z) - f(x, y, z)) / delta,    
        (f(x, y + delta, z) - f(x, y, z)) / delta, 
        (f(x, y, z + delta) - f(x, y, z)) / delta,    
    ])

def partial_derivative(f: callable) -> list:
    return np.array([
        lambda x, y, z: (f(x + delta, y, z) - f(x, y, z)) / delta,
        lambda x, y, z: (f(x, y + delta, z) - f(x, y, z)) / delta,
        lambda x, y, z: (f(x, y, z + delta) - f(x, y, z)) / delta,
    ])

def f(x, y, z):
    return x **2 + y - 3 * z

f_x = partial_derivative(f)[0]
f_xx = partial_derivative(f_x)[0]
f_xy = partial_derivative(f_x)[1]

print(f_x(1,1, 1))
print(f_xx(1, 1, 1))
print(f_xy(1, 1, 1))