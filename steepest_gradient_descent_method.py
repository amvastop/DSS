import numpy as np
from Sven_method import sven_method
from dichotomy_method import dichotomy_method

def steepest_gradient_descent_method(x, e1, e2, M, f, df):
    k = 0
    g_f = df(x)
    while np.linalg.norm(g_f, ord=2)  >= e1 and k < M:
        t = f_t(x, g_f, f)
        tmp = x
        x = x - t * g_f
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
    tmp = [(f(x[0]) - f(x[1])) / (2 * dx) for x in array_x]
    return np.array(tmp)

def f_t(x, g_f, f ):
    t = lambda t: f(x - g_f * t)
    E = 0.000001
    l = 0.00001
    x0 = 0 
    h = 1
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
    x0 =[0.5, 1.0]
    x0 = np.array(x0)
    e1 = 0.1
    e2 = 0.15
    M = 1
    ans = steepest_gradient_descent_method(x0, e1, e2, M, f, df)
    print(ans)
