import scipy as sp
import copy
from Utils.evolutionary_operators import cxUniform, mutateGaussian, binary_tournament_selection
import numpy as np


def RKGA(f_surr, dim, pop, gen, f_con = None, binary_vars=None, verbose=False,
             mutation_stdev=None, mutation_damping=None):
    """
    A canonical evolutionary algorithm with mixed integer handling capacity
    By default, this algorithm encodes all variables in the range [0,1]. Decoding step is thus needed during evaluations
    By default, this algorithm assumes objective function maximization
    """
    ############################ List of tunable hyper-parameters in the Random-key GA ##########################
    if mutation_stdev is None:
        mutation_stdev = 0.03  # mutation perturbation
    if mutation_damping is None:
        mutation_damping = 1  # allows mutation strength to dampen with increasing generations, in order to encourage convergence
    #############################################################################################################

    num_binary_vars, prob_bitflip = process_binary_vars(binary_vars, dim)

    if pop % 2 == 1:
        pop += 1

    P = [sp.rand(dim) for _ in range(pop)]
    if binary_vars is not None:
        P = binarize(P, pop, binary_vars)

    bestFitness = None
    halfpop = int(pop / 2)
    penalty = 1e9  # arbitrarily large multiplier for penalizing constraint violations

    for G in range(gen):

        if f_con is None:
            # fitnesses = [f_surr(x) for x in P]
            fitnesses = f_surr(np.array(P)).tolist()
        else:
            # fitnesses = [f_surr(x) - penalty*f_con(x) for x in P]
            fitnesses = (f_surr(np.array(P)) - penalty*f_con(np.array(P))).tolist()

        maxFitness = max(fitnesses)
        if bestFitness is None:
            bestFitness = maxFitness
            bestSolution = copy.deepcopy(P[sp.argmax(fitnesses)])
            flag = True
        elif maxFitness > bestFitness:
            bestFitness = maxFitness
            bestSolution = copy.deepcopy(P[sp.argmax(fitnesses)])
            flag = True
        else:
            flag = False

        parent_solutions, parent_fitnesses = binary_tournament_selection(pop, P, fitnesses)
        # Incorporate some elitism
        if flag is False:
            parent_solutions[0] = bestSolution
            parent_fitnesses[0] = bestFitness

        if verbose: print("Step", G, "; best fitness found:", bestFitness)

        if G == gen-1:
            break

        # Offspring creation via uniform crossovers OR mutation
        P = []
        for i in range(halfpop):
            offspring = cxUniform(parent_solutions[i], parent_solutions[i + halfpop], dim, 0.5)
            offspring = mutateGaussian(offspring[0], offspring[1], dim, mutation_stdev,
                                    num_binary_vars=num_binary_vars, binary_vars=binary_vars,
                                    prob_bitflip=prob_bitflip)

            P.extend([offspring[0], offspring[1]])

        mutation_stdev = mutation_damping * mutation_stdev

    return bestSolution


def process_binary_vars(binary_vars,dim):
    if binary_vars is not None:
        num_binary_vars = len(binary_vars)
        prob_bitflip = 1 / num_binary_vars
        if num_binary_vars > dim or max(binary_vars) >= dim or len(set(binary_vars)) != num_binary_vars:
            raise Exception('Binary variables must all be unique and should not exceed dimensionality of the problem.')
    else:
        num_binary_vars = None
        prob_bitflip = None

    return num_binary_vars, prob_bitflip


def binarize(x, sample_size, binary_vars):
    for i in range(sample_size):
        subvars = x[i][binary_vars]
        subvars[subvars < 0.5], subvars[subvars >= 0.5] = 0., 1.
        x[i][binary_vars] = copy.deepcopy(subvars)

    return x