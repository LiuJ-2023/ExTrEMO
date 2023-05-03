# This code contains the evolutionary search operators used by the MOEAs in MOEA.py

import random
import math
import copy


def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i



def index(non_dom_sol, min_distance, c_distance):
    for i in range(0, len(c_distance)):
        if c_distance[i] == min_distance:
            return non_dom_sol[i]



def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list



def sort_distance(non_dom_sol, c_distance):
    sorted_distance = []
    while len(sorted_distance) != len(c_distance):
        sorted_distance.append(index(non_dom_sol, min(c_distance), c_distance))
        c_distance[index_of(min(c_distance), c_distance)] = math.inf
    return sorted_distance



def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front



def fast_non_dominated_sort_3D(f1, f2, f3):
    S = [[] for i in range(0, len(f1))]
    front = [[]]
    n = [0 for i in range(0, len(f1))]
    rank = [0 for i in range(0, len(f1))]

    for p in range(0, len(f1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(f1)):
            if (f1[p] < f1[q] and f2[p] < f2[q] and f3[p] < f3[q]) or (f1[p] <= f1[q] and f2[p] < f2[q] and f3[p] < f3[q]) or (f1[p] < f1[q] and f2[p] <= f2[q] and f3[p] < f3[q]) \
                    or (f1[p] < f1[q] and f2[p] < f2[q] and f3[p] <= f3[q]) or (f1[p] <= f1[q] and f2[p] <= f2[q] and f3[p] < f3[q]) \
                    or (f1[p] <= f1[q] and f2[p] < f2[q] and f3[p] <= f3[q]) or (f1[p] < f1[q] and f2[p] <= f2[q] and f3[p] <= f3[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (f1[q] < f1[p] and f2[q] < f2[p] and f3[q] < f3[p]) or (f1[q] <= f1[p] and f2[q] < f2[p] and f3[q] < f3[p]) or (f1[q] < f1[p] and f2[q] <= f2[p] and f3[q] < f3[p]) \
                    or (f1[q] < f1[p] and f2[q] < f2[p] and f3[q] <= f3[p]) or (f1[q] <= f1[p] and f2[q] <= f2[p] and f3[q] < f3[p]) \
                    or (f1[q] <= f1[p] and f2[q] < f2[p] and f3[q] <= f3[p]) or (f1[q] < f1[p] and f2[q] <= f2[p] and f3[q] <= f3[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front



def crowding_distance(values1, values2, front):
    front.sort()
    distance = [0 for i in range(len(values1))]
    c_distance = []
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[sorted1[0]] = 999999999
    distance[sorted1[len(front) - 1]] = 999999999
    for k in range(1, len(front) - 1):
        distance[sorted1[k]] = distance[sorted1[k]] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                    max(values1) - min(values1) + 0.001)
    for k in range(1, len(front) - 1):
        distance[sorted2[k]] = distance[sorted2[k]] + (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2) + 0.001)

    for j in distance:
        if j == 0:
            continue
        else:
            c_distance.append(j)
    return c_distance



def crowding_distance_3D(f1, f2, f3, front):
    front.sort()
    distance = [0 for i in range(len(f1))]
    c_distance = []
    sorted1 = sort_by_values(front, f1[:])
    sorted2 = sort_by_values(front, f2[:])
    sorted3 = sort_by_values(front, f3[:])
    distance[sorted1[0]] = 999999999
    distance[sorted1[len(front) - 1]] = 999999999
    distance[sorted2[0]] = 999999999
    distance[sorted2[len(front) - 1]] = 999999999
    distance[sorted3[0]] = 999999999
    distance[sorted3[len(front) - 1]] = 999999999
    for k in range(1, len(front) - 1):
        distance[sorted1[k]] = distance[sorted1[k]] + (f1[sorted1[k + 1]] - f1[sorted1[k - 1]]) / (max(f1) - min(f1) + 0.001) + 0.001
    for k in range(1, len(front) - 1):
        distance[sorted2[k]] = distance[sorted2[k]] + (f2[sorted2[k + 1]] - f2[sorted2[k - 1]]) / (max(f2) - min(f2) + 0.001) + 0.001
    for k in range(1, len(front) - 1):
        distance[sorted3[k]] = distance[sorted3[k]] + (f3[sorted2[k + 1]] - f3[sorted2[k - 1]]) / (max(f3) - min(f3) + 0.001) + 0.001

    for j in distance:
        if j == 0:
            continue
        else:
            c_distance.append(j)
    return c_distance



def SBX_crossover(parent1, parent2): # With uniform crossover-like variable swap
    child1 = [0 for i in range(len(parent1))]
    child2 = [0 for i in range(len(parent2))]
    lower_bound = 0
    upper_bound = 1
    eta = 20

    for i in range(len(parent1)):
        mew = random.random()
        if mew <= 0.5:
            beta = 2 * mew
        else:
            beta = 1. / (2 * (1 - mew))
        beta **= 1. / (eta + 1)

        c1 = 0.5 * (((1 + beta) * parent1[i]) + ((1 - beta) * parent2[i]))
        if c1 < lower_bound:
            child1[i] = lower_bound
        elif c1 > upper_bound:
            child1[i] = upper_bound
        else:
            child1[i] = c1

        c2 = 0.5 * (((1 - beta) * parent1[i]) + ((1 + beta) * parent2[i]))
        if c2 < lower_bound:
            child2[i] = lower_bound
        elif c2 > upper_bound:
            child2[i] = upper_bound
        else:
            child2[i] = c2

        # Variable swap
        if random.random() <= 0.5:
            temp = copy.deepcopy(child1[i])
            child1[i]=copy.deepcopy(child2[i])
            child2[i]=copy.deepcopy(temp)

    return child1, child2



def polynomial_mutation(parent1, parent2):
    child1 = [0 for i in range(len(parent1))]
    child2 = [0 for i in range(len(parent2))]
    lower_bound = 0
    upper_bound = 1
    eta = 20
    mutation_rate = 1 / len(parent1)

    for i in range(len(parent1)):
        rand = random.random()

        if rand <= mutation_rate:
            u = random.random()
            mut_pow = 1 / (1 + eta)

            delta1_c1 = (parent1[i] - lower_bound) / (upper_bound - lower_bound)
            delta1_c2 = (parent2[i] - lower_bound) / (upper_bound - lower_bound)
            delta2_c1 = (upper_bound - parent1[i]) / (upper_bound - lower_bound)
            delta2_c2 = (upper_bound - parent2[i]) / (upper_bound - lower_bound)

            if u <= 0.5:

                xy1 = 1 - delta1_c1
                xy2 = 1 - delta1_c2
                val1 = (2 * u) + ((1 - (2 * u)) * pow(xy1, 1 + eta))
                val2 = (2 * u) + ((1 - (2 * u)) * pow(xy2, 1 + eta))
                delta_q1 = pow(val1, mut_pow) - 1
                delta_q2 = pow(val2, mut_pow) - 1
                c1 = parent1[i] + (delta_q1 * (upper_bound - lower_bound))
                c2 = parent2[i] + (delta_q2 * (upper_bound - lower_bound))
                child1[i] = min(max(lower_bound, c1), upper_bound)
                child2[i] = min(max(lower_bound, c2), upper_bound)

            else:
                xy1 = 1 - delta2_c1
                xy2 = 1 - delta2_c2
                val1 = (2 * (1 - u)) + (2 * (u - 0.5) * (pow(xy1, 1 + eta)))
                val2 = (2 * (1 - u)) + (2 * (u - 0.5) * (pow(xy2, 1 + eta)))
                delta_q1 = 1 - pow(val1, mut_pow)
                delta_q2 = 1 - pow(val2, mut_pow)
                c1 = parent1[i] + (delta_q1 * (upper_bound - lower_bound))
                c2 = parent2[i] + (delta_q2 * (upper_bound - lower_bound))
                child1[i] = min(max(lower_bound, c1), upper_bound)
                child2[i] = min(max(lower_bound, c2), upper_bound)

        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2



def binary_tournament(index1, index2, f1_values, f2_values):
    if (f1_values[index1] < f1_values[index2] and f2_values[index1] < f2_values[index2]) or (
            f1_values[index1] <= f1_values[index2] and f2_values[index1] < f2_values[index2]) or (
            f1_values[index1] < f1_values[index2] and f2_values[index1] <= f2_values[index2]):
        return index1
    elif (f1_values[index1] > f1_values[index2] and f2_values[index1] > f2_values[index2]) or (
            f1_values[index1] >= f1_values[index2] and f2_values[index1] > f2_values[index2]) or (
            f1_values[index1] > f1_values[index2] and f2_values[index1] >= f2_values[index2]):
        return index2
    else:
        r = random.random()
        if r < 0.5:
            return index1
        else:
            return index2



def binary_tournament_3D(p1, p2, f1, f2, f3):
    if (f1[p1] < f1[p2] and f2[p1] < f2[p2] and f3[p1] < f3[p2]) or (f1[p1] <= f1[p2] and f2[p1] < f2[p2] and f3[p1] < f3[p2]) or (f1[p1] < f1[p2] and f2[p1] <= f2[p2] and f3[p1] < f3[p2]) \
            or (f1[p1] < f1[p2] and f2[p1] < f2[p2] and f3[p1] <= f3[p2]) or (f1[p1] <= f1[p2] and f2[p1] <= f2[p2] and f3[p1] < f3[p2]) \
            or (f1[p1] <= f1[p2] and f2[p1] < f2[p2] and f3[p1] <= f3[p2]) or (f1[p1] < f1[p2] and f2[p1] <= f2[p2] and f3[p1] <= f3[p2]):
        return p1
    elif (f1[p1] > f1[p2] and f2[p1] > f2[p2] and f3[p1] > f3[p2]) or (f1[p1] >= f1[p2] and f2[p1] > f2[p2] and f3[p1] > f3[p2]) or (f1[p1] > f1[p2] and f2[p1] >= f2[p2] and f3[p1] > f3[p2]) \
            or (f1[p1] > f1[p2] and f2[p1] > f2[p2] and f3[p1] >= f3[p2]) or (f1[p1] >= f1[p2] and f2[p1] >= f2[p2] and f3[p1] > f3[p2]) \
            or (f1[p1] >= f1[p2] and f2[p1] > f2[p2] and f3[p1] >= f3[p2]) or f1[p1] > f1[p2] and f2[p1] >= f2[p2] and f3[p1] >= f3[p2]:
        return p2
    else:
        r = random.random()
        if r < 0.5:
            return p1
        else:
            return p2



def check_bounds(x):
    offspring = [0 for _ in range(len(x))]
    lower_bound = 0
    upper_bound = 1

    for i in range(0, len(x)):
        if x[i] < lower_bound:
            offspring[i] = lower_bound
        elif x[i] > upper_bound:
            offspring[i] = upper_bound
        else:
            offspring[i] = x[i]

    return offspring
