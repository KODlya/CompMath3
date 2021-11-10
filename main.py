import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import timeit


def f1(x):
    return np.arctan((x * x)) / x


def f2(x):
    return np.sqrt((1 - 0.7 * x * x) / 2)


# исходная функция 1
def func1(x, y):
    return np.tan(x * y) - x * x


# исходная функция 2
def func2(x, y):
    return 0.7 * x * x + 2 * y * y - 1


# производная по X 1
def func11(x, y):
    return y/(np.cos(x * y)*np.cos(x * y))- 2 * x


# производная по Y 1
def func12(x, y):
    return x / (np.cos(x * y)*np.cos(x * y))


# производная по X 2
def func21(x, y):
    return 1.4 * x


# производная по Y 2
def func22(x, y):
    return 4 * y


def alpha(x, y):
    return 1 / (func12(x, y) * func21(x, y) / func22(x, y) - func11(x, y))


def beta(x, y):
    return func12(x, y) / (func11(x, y) * func22(x, y) - func12(x, y) * func21(x, y))


def gamma(x, y):
    return 1 / (func11(x, y) * func22(x, y) / func21(x, y) - func12(x, y))


def delta(x, y):
    return func11(x, y) / (func21(x, y) * func12(x, y) - func11(x, y) * func22(x, y))


def func(p):
    x, y = p
    return np.tan(x * y) - x * x, 0.7 * x * x + 2 * y * y - 1


def iter_func1(x, y):
    return x + alpha(x, y) * func1(x, y) + beta(x, y) * func2(x, y)


def iter_func2(x, y):
    return y + gamma(x, y) * func1(x, y) + delta(x, y) * func2(x, y)


def newton_func1(x, y):
    return x - np.linalg.det(A1(x, y)) / np.linalg.det(jacobi(x, y))


def newton_func2(x, y):
    return y - np.linalg.det(A2(x, y)) / np.linalg.det(jacobi(x, y))


def A1(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func1(x, y)
    matrix[1][0] = func2(x, y)
    matrix[0][1] = func12(x, y)
    matrix[1][1] = func22(x, y)
    return matrix


def A2(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func11(x, y)
    matrix[1][0] = func21(x, y)
    matrix[0][1] = func1(x, y)
    matrix[1][1] = func2(x, y)
    return matrix


def jacobi(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func11(x, y)
    matrix[1][0] = func21(x, y)
    matrix[0][1] = func12(x, y)
    matrix[1][1] = func22(x, y)
    return matrix


def iteration(x, y, iter_f1, iter_f2, eps):
    count = 0
    while math.sqrt((iter_f1(x, y) - x) ** 2 + (iter_f2(x, y) - y) ** 2) >= eps:
        count += 1
        x = iter_f1(x, y)
        y = iter_f2(x, y)
    return x, y, count


def seidel(x, y, iter_f1, iter_f2, eps):
    count = 0
    while math.sqrt((iter_f1(x, y) - x) ** 2 + (iter_f2(iter_f1(x, y), y) - y) ** 2) >= eps:
        count += 1
        x = iter_f1(x, y)
        y = iter_f2(iter_f1(x, y), y)
    return x, y, count


def newton(x, y, newton_f1, newton_f2, eps):
    count = 0
    while math.sqrt((newton_f1(x, y) - x) ** 2 + (newton_f2(x, y) - y) ** 2) >= eps:
        count += 1
        x = newton_f1(x, y)
        y = newton_f2(x, y)
    return x, y, count


def result_print(res):
    print("x = %.5f y = %.5f" % (res[0], res[1]))
    print("Number of iteration = %d" % res[2])


e = 10e-5
x0 = -1
y0 = -0.5


print("\nIteration method:")
start_time = timeit.default_timer()
res1 = iteration(x0, y0, iter_func1, iter_func2, e)
result_print(res1)
time = (timeit.default_timer() - start_time) * 1000
print("Runtime: %.3f ms" % time)

print("\nSeidel method:")
start_time = timeit.default_timer()
res2 = seidel(x0, y0, iter_func1, iter_func2, e)
result_print(res2)
time = (timeit.default_timer() - start_time) * 1000
print("Runtime: %.3f ms" % time)

print("\nNewton method:")
start_time = timeit.default_timer()
res3 = newton(x0, y0, newton_func1, newton_func2, e)
result_print(res3)
time = (timeit.default_timer() - start_time) * 1000
print("Runtime: %.3f ms" % time)

X = opt.fsolve(func, (x0, y0))[0]
# print("x = %f y = %f" % (X, f1(X)))

x_gr = np.arange(-0.999, 0, 0.01)
plt.rc('grid', linestyle="--", color="black")
plt.grid(True)
plt.xlabel("X", color="black")
plt.ylabel("Y", color="black")
plt.suptitle("График для отделения корней", color="black")
plt.scatter(X, f1(X), color="black", lw=4)
plt.plot(x_gr, f1(x_gr), lw=2, color="deeppink", label="tg(xy) = x*x")
plt.plot(x_gr, -f2(x_gr), lw=2, color="blue", label="0.7*x*x + 2*y*y = 1")

plt.legend()
plt.show()
