# functions for generating weight-vectors and scalarizing multi-objective problems

import numpy as np
from math import factorial as fac
import copy
from Utils import nondomination


def generate_wv(m, H = None, disp_=False):

    if H is None:
        if m == 3: # 3-objective problem
            H = 4
        elif m == 2: # 2-objective problem
            H = 10
        else:
            raise Exception("Code only tested for 2 or 3 objective optimization problems.")

    N = int(fac(H+m-1)/(fac(m-1)*fac(H))) # N = (H+m-1)_C_(m-1)

    if m == 2:
        lamda_ = [np.array([0., 0.]) for _ in range(N)]
        lamda_[0][1] = 1. - lamda_[0][0]
        lamda_[0] = (lamda_[0]+0.0001)/1.0002
        for i in range(N-1):
            lamda_[i+1][0] = lamda_[i][0] + (1.0/H)
            lamda_[i + 1][1] = 1.0 - lamda_[i+1][0]
            lamda_[i + 1] = (lamda_[i + 1] + 0.0001) / 1.0002
    elif m == 3:
        lamda_ = [np.array([0., 0., 0.]) for _ in range(N)]
        count = 0
        for i in range(H + 1):
            for j in range(H + 1 -i):
                lamda_[count][2] = i/H
                lamda_[count][1] = j/H
                lamda_[count][0] = 1.0 - lamda_[count][1] - lamda_[count][2]
                lamda_[count] = (lamda_[count]+0.0001)/1.0003
                count += 1

    if disp_:
        print(lamda_)

    return lamda_ # returns list of weight vectors


def normalize(m, F, disp_=False): # simple but not the most efficient implementation

    if F.shape[1] != m:
        raise Exception('Number of objectives mismatch.')

    # is_efficient = nondomination.is_pareto_efficient_simple(F)
    norm_F = copy.deepcopy(F)

    # F_max = F[is_efficient].max(axis = 0)
    # F_min = F[is_efficient].min(axis = 0)

    # customized for the magnetic sifter problem
    F_max = np.max(norm_F,axis=0)
    F_min = np.min(norm_F,axis=0)
    # mean_F = np.mean(F,axis=0)
    # std_F = np.std(F,axis=0)
    # norm_F = (norm_F - mean_F)/std_F

    for i in range(m):
        norm_F[:,i] = (norm_F[:,i] - F_min[i])/(F_max[i] - F_min[i])
        # norm_F[:,i] = (norm_F[:,i] - mean_F)/std_F

    return norm_F, F_max, F_min # returns objective function values normalizing "PF" to range [0, 1]


def scalarize_tchebycheff(norm_f, lmda, augmented = False):

    scalar_f = np.zeros(norm_f.shape[0])
    for i in range(norm_f.shape[0]):
        scalar_f[i] = np.multiply(norm_f[i],lmda).max()
        if augmented:
            scalar_f[i] = scalar_f[i] + 0.05 * np.dot(lmda, norm_f[i])

    return scalar_f # returns Tchebycheff scalarized objective function values for a given weight vector


# Test
if __name__ == "__main__":

    wv = generate_wv(3, H = 10, disp_ = True)

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    wv = np.array(wv)

    x = wv[:, 0]
    y = wv[:, 1]
    z = wv[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ax.scatter3D(x, y, z, color="blue")

    plt.show()