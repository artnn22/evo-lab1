import numpy as np

from .function import dixon_price
from .ga import evaluation_budget


def run_random_search(config, seed):
    rng = np.random.default_rng(seed)
    best_point = None
    best_score = float("inf")
    evaluations = 0
    history = []

    for generation in range(config["generations"] + 1):
        if generation == 0:
            point_count = config["population_size"]
        else:
            point_count = config["population_size"] - config["elite_size"]

        for i in range(point_count):
            point = []
            for j in range(config["dim"]):
                point.append(rng.uniform(config["bounds"][0], config["bounds"][1]))
            score = dixon_price(point)
            evaluations += 1
            if score < best_score:
                best_score = score
                best_point = point
        history.append(best_score)

    if evaluations != evaluation_budget(config):
        raise RuntimeError("Число вычислений случайного поиска не совпало с бюджетом ГА")

    return {
        "best_x": best_point,
        "best_f": best_score,
        "history": history,
        "n_evals": evaluations,
    }
