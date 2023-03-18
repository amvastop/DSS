import numpy as np

def step_2_3(dots, n):
    f_dots = np.array([f(x) for x in dots])
    x_l = dots[np.argmin(f_dots)]
    index = np.argmax(f_dots)
    x_h = dots[index]
    tmp = np.delete(f_dots, index)
    x_s = dots[np.argmax(tmp)]
    tmp = np.delete(dots, index, 0)
    x_n2 = 1 / n * (np.sum(tmp,  axis=0))
    return f_dots, x_n2, x_l, x_h, x_s, index


def nelder_mead_method(a, b, y, E, n, f, dots=None):
    if dots is None:
        dots = np.random.random((n + 1, n))
        print(dots)
    f_dots, x_n2, x_l, x_h, x_s, index = step_2_3(dots, n)
    tmp = np.sqrt(np.sum((f_dots - f(x_n2)) ** 2) / (n + 1))
    while np.sqrt(np.sum((f_dots - f(x_n2)) ** 2) / (n + 1)) >= E:
        x_n3 = x_n2 + a * (x_n2 - x_h)
        if f(x_n3) <= f(x_l):
            x_n4 = x_n2 + y * (x_n3 - x_n2)
            if f(x_n4) < f(x_l):
                tmp = x_n4
            else:
                tmp = x_n3
            dots[index] = tmp
        elif f(x_s) < f(x_n3) <= f(x_h):
            dots[index] = x_n2 + b * (x_h - x_n2) # x n+5
        elif f(x_l) < f(x_n3) <= f(x_s):
            dots[index] = x_n3
        elif f(x_n3) > f(x_h):
            dots = x_l - 0.5 * (dots - x_l)
        f_dots, x_n2, x_l, x_h, x_s, index = step_2_3(dots, n)
    return x_l



if __name__ == "__main__":
    n = 2
    f = lambda x : 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2
    E = 0.2
    a, b, y = 1, 0.5, 2
    #dots = np.array([[8,9], [10, 11], [8, 11]], dtype=float)
    ans = nelder_mead_method(a, b, y, E, n, f)
    print(f(ans), f((5,6.25)))
