import numpy as np
from tqdm import tqdm

def initialization(N, Dim, UB, LB):
    # B_no = len(UB[0])  # Number of boundaries
    # print(UB)
    if not isinstance(UB[0], list):
        X = (np.random.rand(N, Dim) * (UB[0] - LB[0])) + LB[0]
    else:
        UB = UB[0]
        LB = LB[0]
        X = np.zeros((N, Dim))
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            # print(Lb_i)
            base_i = np.random.rand(N, 1) * (Ub_i - Lb_i)
            base_lb = np.ones((N, 1))*Lb_i
            # print(base_i, base_lb)
            X[:, i] = (base_i + base_lb).squeeze()

    return X


def BKA(pop, T, lb, ub, dim, fobj):
    # print(pop, T, lb, ub, dim, fobj)
    # Initialize the locations of Blue Sheep
    p = 0.9
    XPos = initialization(pop, dim, ub, lb)
    XFit = [fobj(X) for X in XPos]
    Convergence_curve = []

    # Start iteration
    Best_Fitness_BKA = float('inf')
    Best_Pos_BKA = None

    for t in tqdm(range(0, T)):
        sorted_indexes = np.argsort(XFit)
        XLeader_Pos = XPos[sorted_indexes[0]]
        XLeader_Fit = XFit[sorted_indexes[0]]

        for i in range(pop):
            n = 0.05 * np.exp(-2 * (t / T) ** 2)
            r = np.random.rand()
            if p < r:
                XPosNew = XPos[i] + n * (1 + np.sin(r)) * XPos[i]
            else:
                XPosNew = XPos[i] * (n * (2 * r - 1) + 1)

            XPosNew = np.clip(XPosNew, lb, ub)  # Boundary checking

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i] = XPosNew
                XFit[i] = XFit_New

        for i in range(pop):
            r = np.random.rand()
            m = 2 * np.sin(r + np.pi / 2)
            s = np.random.randint(0, pop)
            r_XFitness = XFit[s]
            ori_value = np.random.rand(dim)
            # cauchy_value = np.random.standard_cauchy(dim)
            # cauchy_value = (1/np.pi) * (1/(ori_value**2 + 1))
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            if XFit[i] < r_XFitness:
                XPosNew = XPos[i] + cauchy_value * (XPos[i] - XLeader_Pos)
            else:
                XPosNew = XPos[i] + cauchy_value * (XLeader_Pos - m * XPos[i])

            XPosNew = np.clip(XPosNew, lb, ub)  # Boundary checking

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i] = XPosNew
                XFit[i] = XFit_New

        # Update the optimal Black-winged Kite
        for i in range(pop):
            if XFit[i] < XLeader_Fit:
                Best_Fitness_BKA = min(Best_Fitness_BKA, XFit[i])
                Best_Pos_BKA = XPos[i,:]
            else:
                Best_Fitness_BKA = min(Best_Fitness_BKA, XLeader_Fit)
                Best_Pos_BKA = XLeader_Pos

        # Best_Fitness_BKA = min(Best_Fitness_BKA, XLeader_Fit)
        # Best_Pos_BKA = XLeader_Pos
        Convergence_curve.append(Best_Fitness_BKA)

    return Best_Fitness_BKA, Best_Pos_BKA, Convergence_curve


if __name__ == "__main__":
    
    # Example usage
    def example_obj(inx):
        return np.sum(inx ** 2)  # Example objective function

    pop = 20
    T = 3000
    lb = [-100]
    ub = [100]
    dim = 30
    p = 0.9

    Best_Fit_BKA, Best_Pos_BKA, Convergence_curve = BKA(pop, T, lb, ub, dim, example_obj)
    print("Best Fitness:", Best_Fit_BKA)
