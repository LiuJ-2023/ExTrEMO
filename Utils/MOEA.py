import numpy as np
import random
from Utils.MOEA_operators import SBX_crossover, polynomial_mutation, binary_tournament, crowding_distance, \
    sort_distance, fast_non_dominated_sort, binary_tournament_3D, crowding_distance_3D, fast_non_dominated_sort_3D

# NSGA-II (3 objectives)
def NSGA2_3D(f_obj, max_gen, pop_size, dim, pop_init = None,model_info = None, uncertainty_set = None):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.
    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    function3: third objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    """

    gen_no = 0
    igd_values = []

    n_obj = 3
    # pf_ref = copy.deepcopy(pf_target)
    # pf_ref_len = len(pf_ref)
    # max_obj = [0] * n_obj
    # min_obj = [0] * n_obj
    # for i in range(n_obj):
    #     pf_ref = sorted(pf_ref, key=lambda obj: obj[i])
    #     max_obj[i] = pf_ref[pf_ref_len - 1][i]
    #     min_obj[i] = pf_ref[0][i]
    # for i in range(pf_ref_len):
    #     for j in range(n_obj):
    #         pf_ref[i][j] = (pf_ref[i][j] - min_obj[j]) / (max_obj[j] - min_obj[j])
    # pf_ref_f1 = []
    # pf_ref_f2 = []
    # pf_ref_f3 = []
    # for i in range(pf_ref_len):
    #     pf_ref_f1.append(pf_ref[i][0])
    #     pf_ref_f2.append(pf_ref[i][1])
    #     pf_ref_f3.append(pf_ref[i][2])


    # Initialize search population
    if pop_init is None:
        solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    else:
        solution = pop_init.tolist()
    F = []
    function1_values,function2_values,function3_values = [],[], []
    for i in range(0, pop_size):
        function_values = f_obj(np.array(solution[i])).squeeze()
        F.append(function_values)
        function1_values.append(function_values[0])
        function2_values.append(function_values[1])
        function3_values.append(function_values[2])

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort_3D(function1_values[:], function2_values[:], function3_values[:])
        print("NSGA-II Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        parent_front_f33 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])
            parent_front_f33.append(function3_values[index])

        # Compute IGD values
        # parent_front_f1 = []
        # parent_front_f2 = []
        # parent_front_f3 = []
        # for i in range(len(parent_front_f11)):
        #     parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
        #     parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        #     parent_front_f3.append((parent_front_f33[i] - min_obj[2]) / (max_obj[2] - min_obj[2]))
        # sum_dist = 0
        # for i in range(pf_ref_len):
        #     min_dist = math.inf
        #     for j in range(len(parent_front_f1)):
        #         dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0) \
        #                 + pow(parent_front_f3[j] - pf_ref_f3[i], 2.0)
        #         dist = math.sqrt(dist2)
        #         if dist < min_dist:
        #             min_dist = dist
        #     sum_dist += min_dist
        # igd = sum_dist / pf_ref_len
        # igd_values.append(igd)
        # print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            a1 = random.randint(0, pop_size - 1)
            a2 = random.randint(0, pop_size - 1)
            a = binary_tournament_3D(a1, a2, function1_values[:], function2_values[:], function3_values[:])
            b1 = random.randint(0, pop_size - 1)
            b2 = random.randint(0, pop_size - 1)
            b = binary_tournament_3D(b1, b2, function1_values[:], function2_values[:], function3_values[:])
            c1, c2 = SBX_crossover(solution[a], solution[b])
            c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
            solution2.append(c1_mutated)
            solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        function3_values2 = function3_values[:]
        for i in range(pop_size, 2 * pop_size):
            function_values = f_obj(np.array(solution2[i])).squeeze()
            F.append(function_values)
            function1_values2.append(function_values[0])
            function2_values2.append(function_values[1])
            function3_values2.append(function_values[2])

        non_dominated_sorted_solution2 = fast_non_dominated_sort_3D(function1_values2[:], function2_values2[:], function3_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance_3D(function1_values2[:], function2_values2[:], function3_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values = []
        function3_values = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                function3_values.append(function3_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return solution, np.array(F)