import numpy as np

from .function import clip_to_bounds, dixon_price


def evaluation_budget(config):
    population_size = config["population_size"]
    generations = config["generations"]
    elite_size = config["elite_size"]
    return population_size + generations * (population_size - elite_size)


def tournament_select(population, scores, config, rng):
    winner = rng.integers(len(population))
    for i in range(config["tournament_size"] - 1):
        candidate = rng.integers(len(population))
        if scores[candidate] < scores[winner]:
            winner = candidate
    return population[winner].copy()


def blx_alpha(first_parent, second_parent, config, rng):
    child = []
    alpha = config["blx_alpha"]
    for i in range(len(first_parent)):
        distance = abs(first_parent[i] - second_parent[i])
        left = min(first_parent[i], second_parent[i]) - alpha * distance
        right = max(first_parent[i], second_parent[i]) + alpha * distance
        child.append(rng.uniform(left, right))
    return clip_to_bounds(child, config["bounds"])


def gaussian_mutate(point, config, rng):
    child = point.copy()
    selected = []
    for i in range(len(child)):
        if rng.random() < config["mutation_prob"]:
            selected.append(i)
    for i in selected:
        child[i] += rng.normal(0, config["mutation_sigma"])
    return clip_to_bounds(child, config["bounds"])


def run_ga(config, seed):
    population_size = config["population_size"]
    elite_size = config["elite_size"]
    if elite_size < 1 or elite_size >= population_size:
        raise ValueError("Размер элиты должен быть от 1 до размера популяции минус 1")

    rng = np.random.default_rng(seed)
    population = []
    scores = []
    for i in range(population_size):
        point = []
        for j in range(config["dim"]):
            point.append(rng.uniform(config["bounds"][0], config["bounds"][1]))
        population.append(point)
        scores.append(dixon_price(point))

    evaluations = population_size
    history = [min(scores)]
    for generation in range(config["generations"]):
        ranking = []
        for i in range(population_size):
            ranking.append([scores[i], i])
        ranking.sort()

        new_population = []
        new_scores = []
        for i in range(elite_size):
            index = ranking[i][1]
            new_population.append(population[index].copy())
            new_scores.append(scores[index])

        while len(new_population) < population_size:
            first_parent = tournament_select(population, scores, config, rng)
            if rng.random() < config["crossover_prob"]:
                second_parent = tournament_select(population, scores, config, rng)
                child = blx_alpha(first_parent, second_parent, config, rng)
            else:
                child = first_parent
            child = gaussian_mutate(child, config, rng)
            new_population.append(child)
            new_scores.append(dixon_price(child))
            evaluations += 1

        population = new_population
        scores = new_scores
        history.append(min(scores))

    best_index = scores.index(min(scores))
    return {
        "best_x": population[best_index],
        "best_f": scores[best_index],
        "history": history,
        "n_evals": evaluations,
    }
