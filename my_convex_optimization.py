import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog

def print_a_function(f, values):
    res = minimize_scalar(f, method = " brent ")
    print(" x_min: .02f, f(x_min): .02f " , (res.x, res.fun))

    x = np.linspace(res.x - 1, res.x + 1, values)
    y = [f(val) for val in x]
    plt.plot(x, y, color="blue", label="f")

    plt.scatter(res.x, res.fun, color = "red", marker = "x", label = "Minimum")

    plt.grid()
    plt.legend(loc = 1)

def find_root_bisection(f, min, max):
    if f(min) * f(max) > 0:
        return 0
    c = (min + max)/2
    while abs(f(c)) > 0.001:
        if f(min) * f(c) < 0:
            max = c
        else:
            min = c
        c = (min + max)/2
    return c

def find_root_newton_raphson(f, f_prime, a):
    while abs(f(a)) > 0.001:
        a = a - f(a)/f_prime(a)
        print(a)
    return a

line = lambda x: x**2 - 1
line_p = lambda x: 2*x

find_root_newton_raphson(line, line_p, 5)

def gradient_descent(f, f_prime, start, learning_rate = 0.1):
    old = start
    new = old - learning_rate * f_prime(old)

    while  abs(old - new ) > 0.0001:
        old = new 
        new = new - learning_rate * f_prime(new)
        print(new)
    return new

f = lambda x : (x - 1) ** 4 + x ** 2
f_prime = lambda x : 4*((x-1)**3) + 2*x
start = -1
x_min = gradient_descent(f, f_prime, start, 0.01)
f_min = f(x_min)

print(" xmin: 0.2f, f(x_min): 0.2f " , (x_min, f_min))

def solve_linear_problem(A, b, c):
    res = linprog(c, A, b)
    return round(res.fun), res.x

A  = np.array([[2,1], [4,5],[1, -2]])
b = np.array([10,8,3])
c = np.array([-1,-2])
x = (0, None)
y = (0, None)