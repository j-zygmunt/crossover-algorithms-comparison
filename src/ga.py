import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from datetime import datetime
from itertools import repeat
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import config
from utils import eval_decorator


r""" This code inspired by DEAP`s :func:`varOr` ad :func:`varAnd`.
Part of an evolutionary algorithm applying crossover or reproduction and mutation.
The modified individuals have their fitness invalidated.
The individuals are cloned so returned population is independent of the input population.
"""
def var(population, toolbox, number_of_children, cxpb, mutpb):
    offspring = []
    while len(offspring) <= number_of_children:
        if random.random() < cxpb:
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            children = toolbox.mate(ind1, ind2)
            if children is not None:
                if type(children) is tuple:
                    for child in children:
                        del child.fitness.values
                        offspring.append(child)
                else:
                    del children.fitness.values
                    offspring.append(children)
        else:
            offspring += random.sample(population, 2)

    offspring += population

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


r""" This code inspired by DEAP`s eaMuPlusLambda.
Algorithm is extended with elite strategy and changed vary :func:`var`.
The rest works like standard :func:`eaMuPlusLambda`.
"""
def eaMuPlusLambda(population, toolbox, pop_size, select_size, cxpb, mutpb, ngen, elite_size,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['best', 'date']

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    record['best'] = halloffame.__getitem__(0).tolist()
    record['date'] = datetime.now()
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Elite strategy
        listElitism = list(map(toolbox.clone, tools.selBest(population, elite_size)))

        offspring = toolbox.select(toolbox.clone(population), k=select_size)
        # Vary the population
        offspring = var(offspring, toolbox, pop_size - select_size, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = offspring + listElitism

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        record['best'] = halloffame.__getitem__(0).tolist()
        record['date'] = datetime.now()
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def mutUniform(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from which to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from which to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.uniform(xl, xu)

    return individual,


def evaluate_ga(cfg: config.GAConfig):

    if cfg.is_min:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    else:
        creator.create("Fitness", base.Fitness, weights=(1.0,))

    creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, cfg.x_min, cfg.x_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=cfg.dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", cfg.fun)
    toolbox.register("mate", cfg.cx, **cfg.cx_params)
    toolbox.register("mutate", tools.mutGaussian, indpb=0.5, sigma=1, mu=0)
    # toolbox.register("mutate", mutUniform, indpb=0.5, low=cfg.x_min, up=cfg.x_max)
    toolbox.register("select", tools.selTournament, tournsize=15)

    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = eaMuPlusLambda(pop,
        toolbox,
        pop_size=cfg.pop_size,
        select_size=cfg.select_size,
        elite_size=cfg.elite_size,
        cxpb=cfg.cxpb,
        mutpb=cfg.mutpb,
        ngen=cfg.max_epoch,
        stats=stats,
        halloffame=hof,
        verbose=None)

    return pop, stats, hof, logbook
