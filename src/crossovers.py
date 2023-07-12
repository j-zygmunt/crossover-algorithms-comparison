import numpy as np
import random


def one_point_crossover(ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    crossover_point = random.randint(1, size - 1)
    ind1[crossover_point:], ind2[crossover_point:] = ind2[crossover_point:], ind1[crossover_point:]

    return ind1, ind2


def multipoint_crossover(ind1: np.ndarray, ind2: np.ndarray, crossover_points_amount: int) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(ind1), len(ind2))
    crossover_points = sorted(random.sample(range(1, size), crossover_points_amount))
    crossover_points.append(size)

    child1 = ind1.copy()
    child2 = ind2.copy()

    previous_index = 0
    for i in range(0, len(crossover_points)):
        if i % 2 == 1:
            print(ind2[crossover_points[previous_index]:crossover_points[i]])
            child1[crossover_points[previous_index]:crossover_points[i]] = ind2[crossover_points[previous_index]:crossover_points[i]]
            child2[crossover_points[previous_index]:crossover_points[i]] = ind1[crossover_points[previous_index]:crossover_points[i]]
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


if __name__ == '__main__':
    i1 = np.array([11, 12, 13, 14, 15, 16, 17, 18])
    i2 = np.array([21, 22, 23, 24, 25, 26, 27, 28])
    multipoint_crossover(i1, i2, 3)
