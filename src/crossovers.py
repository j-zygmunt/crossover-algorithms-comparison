import numpy as np
import random


def one_point_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    cxp = random.randint(1, size - 1)
    ind1[cxp:], ind2[cxp:] = ind2[cxp:], ind1[cxp:]

    return ind1, ind2


def multipoint_crossover(ind1: np.ndarray, ind2: np.ndarray, cxps_amount: int) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    cxps = sorted(random.sample(range(1, size), cxps_amount))
    cxps.append(size)

    child1 = ind1.copy()
    child2 = ind2.copy()

    previous_index = 0
    for i in range(0, len(cxps)):
        if i % 2 == 1:
            print(ind2[cxps[previous_index]:cxps[i]])
            child1[cxps[previous_index]:cxps[i]] = ind2[cxps[previous_index]:cxps[i]]
            child2[cxps[previous_index]:cxps[i]] = ind1[cxps[previous_index]:cxps[i]]
        previous_index = i

    return child1, child2


def average_crossover(ind1: np.ndarray, ind2: np.ndarray) -> np.ndarray:
    return (ind1 + ind2) / 2


def blend_aplha_crossover(ind1: np.ndarray, ind2: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    d = np.absolute(ind1 - ind2)

    child1 = np.zeros(size)
    child2 = np.zeros(size)

    for i in range (0, size):
        child1[i] = random.uniform(min(ind1[i], ind2[i]) - alpha * d[i], max(ind1[i], ind2[i]) + alpha * d[i])
        child2[i] = random.uniform(min(ind1[i], ind2[i]) - alpha * d[i], max(ind1[i], ind2[i]) + alpha * d[i])
        pass

    return child1, child2


def blend_aplha_beta_crossover(ind1: np.ndarray, ind2: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    d = np.absolute(ind1 - ind2)

    child1 = np.zeros(size)
    child2 = np.zeros(size)

    for i in range (0, size):
        if ind1[i] <= ind2[i]:
            child1[i] = random.uniform(ind1[i] - alpha * d[i], ind2[i] + beta * d[i])
            child2[i] = random.uniform(ind1[i] - alpha * d[i], ind2[i] + beta * d[i])
        else:
            child1[i] = random.uniform(ind1[i] - beta * d[i], ind2[i] + alpha * d[i])
            child2[i] = random.uniform(ind1[i] - beta * d[i], ind2[i] + alpha * d[i])

    return child1, child2


def uniform_crossover(ind1: np.ndarray, ind2: np.ndarray, swap_propability: float) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))

    child1 = ind1.copy()
    child2 = ind2.copy()

    for i in range (0, size):
        alpha = random.uniform(0, 1)

        if alpha <= swap_propability:
            child1[i] = ind2[i]
            child2[i] = ind1[i]

    return child1, child2

def discrete_crossover(ind1: np.ndarray, ind2: np.ndarray) -> np.ndarray:
    size = min(len(ind1), len(ind2))

    child = ind2.copy()

    for i in range (0, size):
        alpha = random.uniform(0, 1)

        if alpha <= 0.5:
            child[i] = ind1[i]

    return child


def simple_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    cxp = cxp = random.randint(1, size - 1)

    child1 = ind1.copy()
    child2 = ind2.copy()

    alpha = random.uniform(0, 1)

    child1[cxp + 1:] = alpha * ind2[cxp + 1:] + (1 - alpha) * ind1[cxp + 1:]
    child2[cxp + 1:] = alpha * ind1[cxp + 1:] + (1 - alpha) * ind2[cxp + 1:]

    return child1, child2


def arithmetical_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = random.uniform(0, 1)

    child1 = alpha * ind1 + (1 - alpha) * ind2
    child2 = alpha * ind2 + (1 - alpha) * ind1

    return child1, child2


def continious_uniform_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_boundary = np.min(np.maximum(ind1, ind2) / (np.maximum(ind1, ind2) - np.minimum(ind1, ind2)))
    upper_boundary = np.max(-np.minimum(ind1, ind2) / (np.maximum(ind1, ind2) - np.minimum(ind1, ind2)))
    alpha = random.uniform(lower_boundary, upper_boundary)

    child1 = alpha * ind1 + (1 - alpha) * ind2
    child2 = alpha * ind2 + (1 - alpha) * ind1

    return child1, child2


def curved_cylinder_crossover(ind1: np.ndarray, ind2: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))

    diff = sum(np.absolute(ind1, ind2))
    if diff <= 0.0001:
        pass
    elif ind1.fitness.values[0] < alpha and ind2.fitness.values[0] < alpha:
        cxp = random.randint(1, size - 1)
        ind1[cxp:], ind2[cxp:] = ind2[cxp:], ind1[cxp:]

        return child1, child2
    elif ind1.fitness.values[0] > alpha or ind2.fitness.values[0] > alpha:
        child1 = (ind1 * ind1.fitness.values[0] + ind2 * ind2.fitness.values[0]) / (ind1.fitness.values[0] + ind2.fitness.values[0])

        return child1


def diverse_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    beta = np.random.choice(np.linspace(0, 1, num=11), size=1)
    cxp = cxp = random.randint(1, size - 1)

    child1 = ind1.copy()
    child2 = ind2.copy()

    child1[cxp] = ind1[cxp] + beta * (ind2[cxp] - ind1[cxp])
    child2[cxp] = min(ind1[cxp], ind2[cxp]) + beta * (max(ind1[cxp], ind2[cxp]) - min(ind1[cxp], ind2[cxp]))

    child1[cxp + 1:] = ind2[cxp + 1:]
    child2[cxp + 1:] = ind1[cxp + 1:]

    return child1, child2


def parent_centric_blx_aplha_crossover(ind1: np.ndarray, ind2: np.ndarray, alpha: float) -> np.ndarray:
    size = min(len(ind1), len(ind2))
    diff = np.absolute(ind1 - ind2)

    child = np.zeros(size)

    lower_boundary = np.minimum(ind1, ind2)
    upper_boundary = np.maximum(ind1, ind2)

    if random.uniform(0, 1) <= 0.5:
        start = np.maximum(lower_boundary, ind1 - alpha * diff)
        end = np.minimum(upper_boundary, ind1 + alpha * diff)
        child = np.random.uniform(start, end, size=size)
    else:
        start = np.maximum(lower_boundary, ind2 - alpha * diff)
        end = np.minimum(upper_boundary, ind2 + alpha * diff)
        child = np.random.uniform(start, end, size=size)

    return child


def inheritance_crossover(ind1: np.ndarray, ind2: np.ndarray) -> np.ndarray:
    size = min(len(ind1), len(ind2))

    f = np.zeros(size - 1)
    g = np.zeros(size - 1)

    for i in range(0, size - 1):
        f[i] = i1[i] / i1[i + 1]
        g[i] = i1[i] / i1[i + 1]

    cxp = cxp = random.randint(1, size - 1)

    child1 = ind1.copy()
    child2 = ind2.copy()

    child1[cpx] = 1 /  g[cpx] * ind1[cpx]
    child2[cpx] = 1 /  f[cpx] * ind2[cpx]

    return child1, child2


def sphere_crossover(ind1: np.ndarray, ind2: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0, 1)

    return np.sqrt(alpha * ind1 ** 2 + (1 - alpha) * ind2 ** 2)


if __name__ == '__main__':
    i1 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
    i2 = np.array([21, 22, 1, 24, 25, 26, 27, 28])

    print(sphere_crossover(i1, i2))
