import random
import numpy as np
import matplotlib.pyplot as plt
from Sven_method import sven_method
from dichotomy_method import dichotomy_method

def steepest_gradient_descent_method(x, e1, e2, M, f, plot=False,):
    dx = 0.0000001
    df = lambda x : grad_f(dx, x, f)
    array_dots = []
    k = 0
    g_f = df(x)
    while np.linalg.norm(g_f, ord=2)  >= e1 and k < M:
        t = f_t(x, g_f, f)
        tmp = x
        x = x - t * g_f
        if plot:
            array_dots.append(x.copy())
        if np.linalg.norm(x - tmp, ord=2) < e2  and abs( f(x) - f(tmp)) < e2:
            break
        k +=1
        g_f = df(x)
    return x
    


def grad_f(dx, x, f):
    array_x = []
    for i in range(len(x)):
        tmp  = x.copy()
        tmp[i] = tmp[i] - dx
        tmp2 = x.copy()
        tmp2[i] = tmp2[i] + dx
        array_x.append([tmp2, tmp])
    array_x = np.array(array_x)
    tmp = [(f(cloum[0]) - f(cloum[1])) / (2 * dx) for cloum in array_x]
    return np.array(tmp)

def f_t(x, g_f, f,):
    t = lambda t: f(x - g_f * t)
    E = 0.000000001
    l = 0.00001
    random.seed( version=2)
    h = random.uniform(0.1, 1000)
    x0 = random.uniform(-1000, 1000)
    segment = sven_method(x0, h, t)
    if segment:
        a, b = segment 
    ans = dichotomy_method(E, l, t, a, b)
    return ans




if __name__ == "__main__":
    n = 2
    dx = 0.005
    f = lambda x : 2 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2  
    df = lambda x : grad_f(dx, x, f)
    x0 = [0.5, 1.0]
    x0 = np.array(x0)
    e1 = 0.1
    e2 = 0.15
    M = 1
    ans = steepest_gradient_descent_method(x0, e1, e2, M, f)
    print(ans)
