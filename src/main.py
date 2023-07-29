import opfunu
from dataclasses import asdict
from datetime import datetime

from utils import eval_decorator
import config
import crossovers
import ga
import database


configs2D = [
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

ccc_configs = [
    config.GAConfig(cx=crossovers.curved_cylinder_crossover),
    config.GAConfig(cx=crossovers.curved_cylinder_crossover, dim = 10),
    config.GAConfig(cx=crossovers.curved_cylinder_crossover, dim = 20)
]

fitness_functions = [
    {'fun':"F22022", 'cx_params':{'alpha': 440.0}},
    {'fun':"F52022", 'cx_params':{'alpha': 990.0}},
]

if __name__ == "__main__":
    # for cfg in configs2D:
    #     for fun in fitness_functions:
    #         func = opfunu.get_functions_by_classname(fun['fun'])[0](ndim=cfg.dim)
    #         cfg.fun = eval_decorator(func.evaluate)

    # for cfg in ccc_configs:
    #     for fun in fitness_functions:
    #         cfg.fun = fun['fun']
    #         if fun['cx_params'] is not None:
    #             cfg.cx_params = fun['cx_params']

    cfg = config.GAConfig(cx=crossovers.one_point_crossover)

    con = database.get_db_connection('db/Experiments.db')
    database.clear_db(con)
    database.prepare_db(con)

    for fun in fitness_functions:
        func = opfunu.get_functions_by_classname(fun['fun'])[0](ndim=cfg.dim)
        cfg.fun = eval_decorator(func.evaluate)
        pop, stats, hof, logbook = ga.evaluate_ga(cfg)

        date = datetime.now()
        cfg_str = str(config.config_asdict(cfg))
        for log in logbook:
            log['experiment_id'] = 1
            log['cx'] = cfg.cx.__name__
            log['fun'] = fun['fun']
            log['dim'] = cfg.dim
            log['date'] = date
            log['config'] = cfg_str
            log['best'] = str(log['best'])

            database.insert_experiment_data(con, log)
