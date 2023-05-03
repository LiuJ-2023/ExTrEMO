import scipy as sp
import numpy as np
from random import sample
import copy

def cxUniform(ind1, ind2, dim, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param dim: dimensionality of individuals
    :param indpb: Independent probability for each attribute to be exchanged.
    """
    rvec = sp.rand(dim)
    boolvec = rvec < indpb
    temp = sp.zeros(dim)
    temp[boolvec] = copy.deepcopy(ind1[boolvec])
    ind1[boolvec] = copy.deepcopy(ind2[boolvec])
    ind2[boolvec] = copy.deepcopy(temp[boolvec])

    # for i in range(dim):
    #     if random() < indpb:
    #         ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def mutateGaussian(ind1, ind2, dim, mutation_stdev, num_binary_vars = None,binary_vars = None, prob_bitflip = None):
    
    mutvec1 = mutation_stdev * sp.randn(dim)
    mutvec2 = mutation_stdev * sp.randn(dim)

    ind1_tmp = ind1 + mutvec1
    ind2_tmp = ind2 + mutvec2

    if num_binary_vars is not None:
        ind1_tmp[binary_vars] = ind1[binary_vars]
        ind2_tmp[binary_vars] = ind2[binary_vars]
        ind1, ind2 = mutateBitFlip(ind1_tmp, ind2_tmp, num_binary_vars, binary_vars, prob_bitflip)
    else:
        ind1 = copy.deepcopy(ind1_tmp)
        ind2 = copy.deepcopy(ind2_tmp)

    ind1[ind1 > 1] = 1
    ind1[ind1 < 0] = 0

    ind2[ind2 > 1] = 1
    ind2[ind2 < 0] = 0

    return ind1, ind2


def mutateBitFlip(ind1, ind2, num_binary_vars, binary_vars, prob_bitflip):

    rvec = sp.rand(num_binary_vars)
    boolvec = rvec < prob_bitflip
    subvars = ind1[binary_vars]
    subvars[boolvec] = 1 - subvars[boolvec]
    ind1[binary_vars] = copy.deepcopy(subvars)

    rvec = sp.rand(num_binary_vars)
    boolvec = rvec < prob_bitflip
    subvars = ind2[binary_vars]
    subvars[boolvec] = 1 - subvars[boolvec]
    ind2[binary_vars] = copy.deepcopy(subvars)

    return ind1, ind2


def binary_tournament_selection(pop, P, fitnesses):

    reshuffle_indx = np.random.permutation(pop)
    parent_solutions = []
    parent_fitnesses = []
    halfpop = int(pop/2)
    for i in range(halfpop):
        if fitnesses[reshuffle_indx[i]] >= fitnesses[reshuffle_indx[i + halfpop]]:
            parent_solutions.append(P[reshuffle_indx[i]])
            parent_fitnesses.append(fitnesses[reshuffle_indx[i]])
        else:
            parent_solutions.append(P[reshuffle_indx[i + halfpop]])
            parent_fitnesses.append(fitnesses[reshuffle_indx[i + halfpop]])
            
    reshuffle_indx = np.random.permutation(pop)
    for i in range(halfpop):
        if fitnesses[reshuffle_indx[i]] >= fitnesses[reshuffle_indx[i + halfpop]]:
            parent_solutions.append(P[reshuffle_indx[i]])
            parent_fitnesses.append(fitnesses[reshuffle_indx[i]])
        else:
            parent_solutions.append(P[reshuffle_indx[i + halfpop]])
            parent_fitnesses.append(fitnesses[reshuffle_indx[i + halfpop]])
    return parent_solutions, parent_fitnesses