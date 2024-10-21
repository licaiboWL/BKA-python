import numpy as np
from numpy import prod


def CEC2005(F):
    if F == 'F1':
        fobj = F1
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F2':
        fobj = F2
        lb = -10
        ub = 10
        dim = 30
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F5':
        fobj = F5
        lb = -30
        ub = 30
        dim = 30
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F7':
        fobj = F7
        lb = -1.28
        ub = 1.28
        dim = 30
    elif F == 'F8':
        fobj = F8
        lb = -500
        ub = 500
        dim = 30
    elif F == 'F9':
        fobj = F9
        lb = -5.12
        ub = 5.12
        dim = 30
    elif F == 'F10':
        fobj = F10
        lb = -32
        ub = 32
        dim = 30
    elif F == 'F11':
        fobj = F11
        lb = -600
        ub = 600
        dim = 30
    elif F == 'F12':
        fobj = F12
        lb = -50
        ub = 50
        dim = 30
    elif F == 'F13':
        fobj = F13
        lb = -50
        ub = 50
        dim = 30
    elif F == 'F14':
        fobj = F14
        lb = -65.536
        ub = 65.536
        dim = 2
    elif F == 'F15':
        fobj = F15
        lb = -5
        ub = 5
        dim = 4
    elif F == 'F16':
        fobj = F16
        lb = -5
        ub = 5
        dim = 2
    elif F == 'F17':
        fobj = F17
        lb = [-5, 0]
        ub = [10, 15]
        dim = 2
    elif F == 'F18':
        fobj = F18
        lb = -2
        ub = 2
        dim = 2
    elif F == 'F19':
        fobj = F19
        lb = 0
        ub = 1
        dim = 3
    elif F == 'F20':
        fobj = F20
        lb = 0
        ub = 1
        dim = 6
    elif F == 'F21':
        fobj = F21
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F22':
        fobj = F22
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F23':
        fobj = F23
        lb = 0
        ub = 10
        dim = 4
    else:
        raise ValueError(f"Unknown function: {F}")

    return fobj, lb, ub, dim


# F1
def F1(x):
    return np.sum(x ** 2)


# F2
def F2(x):
    return np.sum(abs(x)) + prod(abs(x))


# F3
def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += np.sum(x[:i + 1]) ** 2
    return o


# F4
def F4(x):
    return max(abs(x))


# F5
def F5(x):
    dim = len(x)
    o = 0
    for i in range(1, dim):
        o += 100 * (x[i] - x[i - 1] ** 2) ** 2 + (x[i - 1] - 1) ** 2
    return o


# F6
def F6(x):
    return np.sum(abs(x + 0.5) ** 2)


def F7(x):
    dim = x.shape[0]
    return np.sum(np.arange(1, dim + 1) * (x ** 4)) + np.random.rand()

def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

def F10(x):
    dim = x.shape[0]
    return -20 * np.exp(-.2 * np.sqrt(np.sum(x ** 2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)

def F11(x):
    dim = x.shape[0]
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1

# def F12(x):
#     print(x)
#     # dim = x.shape[0]
#     dim = len(x)
#     pi_over_dim = np.pi / dim
#     term1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2
#     term2 = sum(((x[1:dim - 1] + 1) / 4) ** 2 * (1 + 10 * (np.sin(pi_over_dim * (1 + (x[1:dim] + 1) / 4))) ** 2))
#     term3 = ((x[-1] + 1) / 4) ** 2
#     term4 = sum(Ufun(x, 10, 100, 4))
#     return pi_over_dim * (term1 + term2 + term3 + term4)

def F12(x):
    dim = x.shape[0]  # 获取输入向量 x 的维度
    # dim = len(x)
    o = (np.pi / dim) * (
        10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4)))**2) +
        np.sum((((x[:dim-1] + 1) / 4)**2) *
               (1 + 10 * ((np.sin(np.pi * (1 + (x[1:dim] + 1) / 4)))**2)))
        + ((x[-1] + 1) / 4)**2 +
        np.sum(Ufun(x, 10, 100, 4))
    )
    return o
def F13(x):
    dim = x.shape[0]
    term1 = ((np.sin(3 * np.pi * x[0])) ** 2 + sum((x[:dim - 1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:dim])) ** 2)))
    term3 = ((x[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[-1])) ** 2)
    term4 = sum(Ufun(x, 5, 100, 4))
    return 0.1*(term1 + term3) + term4

def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j]) ** 6)
    return (1 / 500 + sum(1 / (np.arange(1, 26) + bS))) ** (-1)

def F15(x):
    aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
    bK = np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16]) ** -1
    return np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)

# F16
def F16(x):
    x = np.array(x)
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)


# F17
def F17(x):
    x = np.array(x)
    return ((x[1] - (x[0] ** 2) * 5.1 / (4 * np.pi ** 2) + 5 / np.pi * x[0] - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)


# F18
def F18(x):
    x = np.array(x)
    return ((1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) *
            (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                        18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)))


def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                   [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])

    o = 0
    for i in range(4):
        aH_i = aH[i]
        cH_i = cH[i]
        pH_i = pH[i]

        diff = x - pH_i
        term = np.sum(aH_i * (diff ** 2))

        # 指数项前面不需要额外的负号
        exp_term = np.exp(-(term))  # 正确地计算指数部分

        o += cH_i * exp_term

    return -o


def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

    o = 0
    for i in range(4):
        aH_i = aH[i]
        cH_i = cH[i]
        pH_i = pH[i]

        diff = x - pH_i
        term = np.sum(aH_i * (diff ** 2))

        # 指数项前面不需要额外的负号
        exp_term = np.exp(-(term))  # 正确地计算指数部分

        o += cH_i * exp_term

    return -o


# # 使用示例：
# x = np.array([0.5, 1.5, 2.5])  # 4维输入向量
# result_F19 = F19(x)
# x = np.array([0.5, 1.5, 2.5, 2.1, 3.7, 4.0])  # 4维输入向量
# result_F20 = F20(x)
#
# print("F19 result:", result_F19)
# print("F20 result:", result_F20)


# F21
def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    x = np.array(x)
    o = 0
    for i in range(10):
        o -= (np.dot((x - aSH[i]), (x - aSH[i])) + cSH[i]) ** (-1)
    return o


# F22
def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    x = np.array(x)
    o = 0
    for i in range(7):
        o -= (np.dot((x - aSH[i]), (x - aSH[i])) + cSH[i]) ** (-1)
    return o


# F23
def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    x = np.array(x)
    o = 0
    for i in range(10):
        o -= (np.dot((x - aSH[i]), (x - aSH[i])) + cSH[i]) ** (-1)
    return o


# Ufun
def Ufun(x, a, k, m):
    x = np.array(x)
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
