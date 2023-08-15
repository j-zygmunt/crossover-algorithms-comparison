import sqlite3
import sys
import os
sys.path.insert(1, os.path.abspath(".."))

import lib.opfunu as opfunu
from utils import eval_decorator
import config
import crossovers
import ga
import database


ITERATIONS = 10
DIMS = [2, 10, 20]
POP_SIZES = [100, 500, 1000]
CCC_CONFIG = config.GAConfig(cx=crossovers.curved_cylinder_crossover)

configs = [
    config.GAConfig(cx=crossovers.one_point_crossover),
    config.GAConfig(cx=crossovers.multipoint_crossover, cx_params={'cxps_amount':5}),
    config.GAConfig(cx=crossovers.uniform_crossover, cx_params={'swap_propability':0.5}),
    config.GAConfig(cx=crossovers.discrete_crossover),
    config.GAConfig(cx=crossovers.average_crossover),
    config.GAConfig(cx=crossovers.blend_alpha_crossover, cx_params={'alpha':0.5}),
    config.GAConfig(cx=crossovers.blend_alpha_beta_crossover, cx_params={'alpha':0.75, 'beta':0.25}),
    config.GAConfig(cx=crossovers.arithmetical_crossover),
    config.GAConfig(cx=crossovers.simple_crossover),
    config.GAConfig(cx=crossovers.diverse_crossover),
    config.GAConfig(cx=crossovers.parent_centric_blx_alpha_crossover, cx_params={'alpha': 0.5}),
    config.GAConfig(cx=crossovers.inheritance_crossover),
    config.GAConfig(cx=crossovers.gene_pooling_crossover),
    config.GAConfig(cx=crossovers.adaptive_probablility_of_gene_crossover),
]

fitness_functions = [
    {'fun':"F12022", 'cx_params':{'alpha': 330.0}},
    {'fun':"F22022", 'cx_params':{'alpha': 440.0}},
    {'fun':"F42022", 'cx_params':{'alpha': 880.0}},
    {'fun':"F52022", 'cx_params':{'alpha': 990.0}},
    {'fun':"F12021", 'cx_params':{'alpha': 110.0}},
    {'fun':"F22021", 'cx_params':{'alpha': 1210.0}},
    {'fun':"F42017", 'cx_params':{'alpha': 440.0}},
    {'fun':"F52017", 'cx_params':{'alpha': 550.0}},
    {'fun':"F12013", 'cx_params':{'alpha': -1540.0}},
    {'fun':"F22013", 'cx_params':{'alpha': -1430.0}},
    {'fun':"F42013", 'cx_params':{'alpha': -1210.0}},
    {'fun':"F52013", 'cx_params':{'alpha': -1100.0}},
    {'fun':"F82013", 'cx_params':{'alpha': -770.0}},
    {'fun':"F92013", 'cx_params':{'alpha': -660.0}},
    {'fun':"F102013", 'cx_params':{'alpha': -550.0}},
    {'fun':"F112013", 'cx_params':{'alpha': -440.0}},
    {'fun':"F132013", 'cx_params':{'alpha': -220.0}},
    {'fun':"F162013", 'cx_params':{'alpha': 220.0}},
    {'fun':"F172013", 'cx_params':{'alpha': 330.0}},
    {'fun':"F202013", 'cx_params':{'alpha': 660.0}},
]


def evaluate(
    experiment_id : int,
    cfg : config.GAConfig,
    con : sqlite3.Connection):

    pop, stats, hof, logbook = ga.evaluate_ga(cfg)
    cfg_str = str(config.config_asdict(cfg))

    for log in logbook:
        log['experiment_id'] = 1
        log['cx'] = cfg.cx.__name__
        log['fun'] = fun['fun']
        log['dim'] = cfg.dim
        log['date'] = str(log['date'])
        log['config'] = cfg_str
        log['best'] = str(log['best'])

        database.insert_experiment_data(con, log)


if __name__ == "__main__":
    con = database.get_db_connection('db/Experiments.db')
    database.clear_db(con)
    database.prepare_db(con)

    experiment_id = 1

    for dim in DIMS:
        for pop_size in POP_SIZES:
            for fitness_function in fitness_functions:
                for _ in range(ITERATIONS):
                    fun = eval_decorator(opfunu.get_functions_by_classname(fitness_function['fun'])[0](ndim=dim).evaluate)
                    for cfg in configs:
                        cfg.dim = dim
                        cfg.pop_size = pop_size
                        cfg.fun = fun
                        evaluate(experiment_id, cfg, con)
                        experiment_id += 1

                    cfg = CCC_CONFIG
                    cfg.dim = dim
                    cfg.pop_size = pop_size
                    cfg.fun = fun
                    if fitness_function['cx_params'] is not None:
                        cfg.cx_params = fitness_function['cx_params']

                    evaluate(experiment_id, cfg, con)
                    experiment_id += 1
