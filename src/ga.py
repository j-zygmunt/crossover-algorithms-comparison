import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import config
from utils import eval_decorator

r""" This code inspired by DEAP`s :func:`varOr` ad :func:`varAnd`.
Part of an evolutionary algorithm applying crossover or reproduction and mutation.
The modified individuals have their fitness invalidated.
The individuals are cloned so returned population is independent of the input population.
"""
def var(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []
    while len(offspring) <= lambda_:
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

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


r""" This code inspired by DEAP`s eaMuPlusLambda.
Algorithm is extended with elite strategy and changed vary :func:`var`.
The rest works like standard :func:`eaMuPlusLambda`.
"""
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, elite,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['best']

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    record['best'] = halloffame.__getitem__(0).tolist()
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Elite strategy
        listElitism = list(map(toolbox.clone, tools.selBest(population, elite)))

        # Vary the population
        offspring = var(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring + listElitism, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        record['best'] = halloffame.__getitem__(0).tolist()
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def evaluate_ga(cfg: config.GAConfig):
    # random.seed(64)

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
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=5)

    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = eaMuPlusLambda(pop,
        toolbox,
        mu=cfg.pop_size,
        lambda_=cfg.pop_size,
        elite=cfg.elite_size,
        cxpb=cfg.cxpb,
        mutpb=cfg.mutpb,
        ngen=cfg.max_epoch,
        stats=stats,
        halloffame=hof,
        verbose=None)

    return pop, stats, hof, logbook
