import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

omega = 2 # right boundary of omega

def k(x):
    # function given in task
    return 1 if x <= 1 else 2 * x


def e(x, i, n):
    # function e_i(x)
    h = omega / n
    if x >= h * (i - 1) and x <= h * i:
        return x/h - i + 1
    if x >= h * i and x <= h * (i + 1):
        return -x/h + i + 1
    return 0

def e_prim(x, i, n):
    # derivative of function e
    h = omega / n
    if x >= h * (i - 1) and x <= h * i:
        return 1 / h
    if x >= h * i and x <= h * (i + 1):
        return -1 / h
    return 0

def B_matrix(n):
    # generating parameters of matrix B
    h = omega / n
    B = np.zeros((n + 1, n + 1))
    for i in range (n): # we leave last row empty because of Dirichlet condition
        for j in range (n + 1):
            boundary_bottom = max(0, h * (i - 1), h * (j - 1))
            boundary_top = min(omega, h * (i + 1), h * (j + 1))
            # we set new integration boundaries in order to avoid integration of 0
            integral = quad(lambda x: k(x) * e_prim(x, i, n) * e_prim(x, j, n), boundary_bottom, boundary_top)[0]
            B[i, j] = integral
    B[0, 0] += 1 # Robin condition
    B[n, n] = 1 # Dirichlet condition
    return B

def L_matrix(n):
    # generating parameters of matrix L
    h = omega / n
    L = np.zeros(n + 1)
    for i in range (n):
        lower_boundary = max(0, h * (i - 1))
        upper_boundary = min(omega, h * (i + 1))
        # again, we set new integration boundaries as above
        integral = 100 * quad(lambda x: x**2 * e(x, i, n), lower_boundary, upper_boundary)[0]
        L[i] = integral
    L[0] -= 20 # Robin condition
    L[n] = -20 # Dirichlet condition
    return L

def fem_solver(n):
    matB = B_matrix(n)
    matL = L_matrix(n)

    space = np.linspace(0, 2, n + 1) # creating nodes
    u: np.ndarray = np.linalg.solve(matB, matL) # solving system of equations created by our matrixes

    # print(matB)
    # print(matL)

    # graphical presentation:
    plt.plot(space, u)
    plt.title("Heat transport equation")
    plt.xlabel("x")
    plt.ylabel("Heat temperature u(x)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    n = int(input("Your n: "))
    fem_solver(n)

