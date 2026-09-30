import json
import os

import numpy as np

from src.function import BOUNDS, DIM, analytical_minimum, clip_to_bounds, dixon_price
from src.ga import blx_alpha, evaluation_budget, gaussian_mutate, run_ga
from src.random_search import run_random_search


def load_config():
    lab_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lab_folder, "configs", "base.json")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def test_function():
    point = analytical_minimum()
    assert len(point) == DIM
    assert point[0] == 1
    assert abs(dixon_price(point)) < 1e-10
    assert abs(dixon_price([1 / 3, 0, 0, 0, 0, 0, 0, 0]) - 2 / 3) < 1e-10


def test_bounds():
    assert clip_to_bounds([-20, 0, 15], BOUNDS) == [-10, 0, 10]
    config = load_config()
    config["mutation_prob"] = 1
    config["mutation_sigma"] = 5
    rng = np.random.default_rng(1)
    point = [9.9] * DIM
    first_parent = [-10] * DIM
    second_parent = [10] * DIM
    for i in range(50):
        point = gaussian_mutate(point, config, rng)
        child = blx_alpha(first_parent, second_parent, config, rng)
        for value in point + child:
            assert BOUNDS[0] <= value <= BOUNDS[1]


def test_search():
    config = load_config()
    config["generations"] = 12
    first = run_ga(config, 1000)
    repeat = run_ga(config, 1000)
    other = run_ga(config, 1001)
    random = run_random_search(config, 1000)
    random_repeat = run_random_search(config, 1000)

    assert first == repeat
    assert first != other
    assert random == random_repeat
    assert first["n_evals"] == evaluation_budget(config)
    assert random["n_evals"] == evaluation_budget(config)
    assert len(first["history"]) == config["generations"] + 1
    assert len(random["history"]) == config["generations"] + 1
    assert first["best_f"] == dixon_price(first["best_x"])
    assert random["best_f"] == dixon_price(random["best_x"])
    for value in first["best_x"] + random["best_x"]:
        assert BOUNDS[0] <= value <= BOUNDS[1]
    for i in range(1, len(first["history"])):
        assert first["history"][i] <= first["history"][i - 1]
        assert random["history"][i] <= random["history"][i - 1]


if __name__ == "__main__":
    test_function()
    test_bounds()
    test_search()
    print("Все проверки пройдены.")
