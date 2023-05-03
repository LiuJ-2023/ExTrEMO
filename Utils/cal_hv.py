from Utils import nondomination, hypervolume
import numpy as np
import copy

def cal_hv(yT,norm_for_hv, ref_point=1.1):
    hv = []
    m,n = yT.shape
    for i in range(20, m):
        Ftemp = yT[:i]
        is_efficient = nondomination.is_pareto_efficient_simple(Ftemp)
        hv.append(hypervolume.hv(copy.deepcopy((Ftemp[is_efficient,:] - norm_for_hv[0])/(norm_for_hv[1]-norm_for_hv[0])), ref_point*np.ones(Ftemp.shape[1])))
    return hv