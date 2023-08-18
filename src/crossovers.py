import numpy as np
import random
import math

def one_point_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    cxp = random.randint(1, size - 1)
    ind1[cxp:], ind2[cxp:] = ind2[cxp:], ind1[cxp:]

    return ind1, ind2


def multipoint_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    cxps_amount: int=2) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    cxps = sorted(random.sample(range(1, size), cxps_amount))
    cxps.append(size)

    parent1 = ind1.copy().astype(float)
    parent2 = ind2.copy().astype(float)

    previous_index = 0
    for i in range(0, len(cxps)):
        if i % 2 == 1:
            ind1[cxps[previous_index]:cxps[i]] = parent1[cxps[previous_index]:cxps[i]]
            ind2[cxps[previous_index]:cxps[i]] = parent2[cxps[previous_index]:cxps[i]]
        previous_index = i

    return ind1, ind2


def uniform_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    swap_propability: float=0.5) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))

    for i in range (0, size):
        if random.uniform(0, 1) <= swap_propability:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def discrete_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    size = min(len(ind1), len(ind2))

    for i in range (0, size):
        if random.uniform(0, 1) > 0.5:
            ind1[i] = ind2[i]

    return ind1


def average_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    size = min(len(ind1), len(ind2))

    for i in range(size):
        ind1[i] = (ind1[i] + ind2[i]) / 2

    return ind1


def blend_alpha_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    alpha: float=0.5) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    d = np.absolute(ind1 - ind2)

    lower_boundary = np.minimum(ind1, ind2) - alpha * d
    upper_boundary = np.maximum(ind1, ind2) + alpha * d

    for i in range(size):
        ind1[i] = np.random.uniform(lower_boundary[i], upper_boundary[i])
        ind2[i] = np.random.uniform(lower_boundary[i], upper_boundary[i])

    return ind1, ind2


def blend_alpha_beta_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    alpha: float=0.75,
    beta: float=0.25) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    d = np.absolute(ind1 - ind2)

    parent1 = ind1.copy().astype(float)
    parent2 = ind2.copy().astype(float)

    for i in range (0, size):
        if parent1[i] <= parent2[i]:
            ind1[i] = random.uniform(parent1[i] - alpha * d[i], parent2[i] + beta * d[i])
            ind2[i] = random.uniform(parent1[i] - alpha * d[i], parent2[i] + beta * d[i])
        else:
            ind1[i] = random.uniform(parent2[i] - beta * d[i], parent1[i] + alpha * d[i])
            ind2[i] = random.uniform(parent2[i] - beta * d[i], parent1[i] + alpha * d[i])

    return ind1, ind2


def arithmetical_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    alpha = random.uniform(0, 1)
    ind1[:], ind2[:] = alpha * ind1 + (1 - alpha) * ind2, alpha * ind2 + (1 - alpha) * ind1

    return ind1, ind2


def simple_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    cxp = random.randint(1, size - 1)

    parent1 = ind1.copy().astype(float)
    parent2 = ind2.copy().astype(float)

    alpha = random.uniform(0, 1)

    ind1[cxp + 1:] = alpha * parent2[cxp + 1:] + (1 - alpha) * parent1[cxp + 1:]
    ind2[cxp + 1:] = alpha * parent1[cxp + 1:] + (1 - alpha) * parent2[cxp + 1:]

    return ind1, ind2


def curved_cylinder_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    alpha: float) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    diff = sum(np.absolute(ind1, ind2))

    if math.isclose(diff, 0.0):
        pass

    elif ind1.fitness.values[0] <= alpha and ind2.fitness.values[0] <= alpha:
        cxp = random.randint(1, size - 1)
        ind1[cxp:], ind2[cxp:] = ind2[cxp:], ind1[cxp:]

        return ind1, ind2

    elif ind1.fitness.values[0] > alpha or ind2.fitness.values[0] > alpha:
        ind1[:] = (ind1 * ind1.fitness.values[0] + ind2 * ind2.fitness.values[0]) / (ind1.fitness.values[0] + ind2.fitness.values[0])

        return ind1


def diverse_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    size = min(len(ind1), len(ind2))
    beta = np.random.choice(np.linspace(0, 1, num=11), size=1)
    cxp = random.randint(1, size - 1)

    u = max(ind1[cxp], ind2[cxp])
    l = min(ind1[cxp], ind2[cxp])

    ind1[cxp] = ind1[cxp] + beta * (ind2[cxp] - ind1[cxp])
    ind2[cxp] = l + beta * (u - l)

    ind1[cxp + 1:], ind2[cxp + 1:] = ind2[cxp + 1:], ind1[cxp + 1:]

    return ind1, ind2


def parent_centric_blx_alpha_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray,
    alpha: float=0.5) -> np.ndarray:

    size = min(len(ind1), len(ind2))
    diff = np.absolute(ind1 - ind2)
    lower_boundary = np.minimum(ind1, ind2)
    upper_boundary = np.maximum(ind1, ind2)

    if random.uniform(0, 1) <= 0.5:
        start = np.maximum(lower_boundary, ind1 - alpha * diff)
        end = np.minimum(upper_boundary, ind1 + alpha * diff)
        ind1[:] = np.random.uniform(start, end, size=size)
    else:
        start = np.maximum(lower_boundary, ind2 - alpha * diff)
        end = np.minimum(upper_boundary, ind2 + alpha * diff)
        ind1[:] = np.random.uniform(start, end, size=size)

    return ind1


# def inheritance_crossover(
#     ind1: np.ndarray,
#     ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

#     size = min(len(ind1), len(ind2))

#     f = np.zeros(size - 1)
#     g = np.zeros(size - 1)

#     for i in range(0, size - 1):
#         f[i] = ind1[i + 1] / ind1[i]
#         g[i] = ind2[i + 1] / ind2[i]

#     np.nan_to_num(f, nan=1, copy=False, posinf=1, neginf=1)
#     np.nan_to_num(g, nan=1, copy=False, posinf=1, neginf=1)

#     cxp = random.randint(0, size - 2)

#     if (not math.isclose(g[cxp] * ind1[cxp], 0.0)) and (not math.isclose(f[cxp] * ind2[cxp], 0.0)):
#         ind1[cxp] = 1 /  g[cxp] * ind1[cxp]
#         ind2[cxp] = 1 /  f[cxp] * ind2[cxp]

#     return ind1, ind2


def fitness_guided_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    size = min(len(ind1), len(ind2))
    alpha = random.uniform(-0.5, 0.5)

    if ind1.fitness < ind2.fitness:
        ind1[:] = ind1 + alpha * (ind1 - ind2)
    else:
        ind1[:] = ind2 + alpha * (ind2 - ind1)

    return ind1


def adaptive_probablility_of_gene_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    size = min(len(ind1), len(ind2))
    xp = ind2.fitness.values[0] / (ind1.fitness.values[0] + ind2.fitness.values[0])

    for i in range (0, size):
        if random.uniform(0, 1) > xp:
            ind1[i] = ind2[i]

    return ind1


def gene_pooling_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    s = np.concatenate([ind1, ind2])
    ind1[:] = np.random.choice(s, int((len(ind1) + len(ind2)) / 2), replace=False)

    return ind1


def sphere_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> np.ndarray:

    alpha = random.uniform(0, 1)
    ind1[:] = np.sqrt(alpha * ind1 ** 2 + (1 - alpha) * ind2 ** 2)

    return ind1


# Do not use for unnormalized data!
def continious_uniform_crossover(
    ind1: np.ndarray,
    ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    lower_boundary = np.min(np.maximum(ind1, ind2) / (np.maximum(ind1, ind2) - np.minimum(ind1, ind2)))
    upper_boundary = np.max(-np.minimum(ind1, ind2) / (np.maximum(ind1, ind2) - np.minimum(ind1, ind2)))
    alpha = random.uniform(lower_boundary, upper_boundary)

    ind1[:], ind2[:] = alpha * ind1 + (1 - alpha) * ind2, alpha * ind2 + (1 - alpha) * ind1

    return ind1, ind2
