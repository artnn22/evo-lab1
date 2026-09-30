import csv
import json
import os
import statistics

import matplotlib.pyplot as plt

from .ga import run_ga
from .random_search import run_random_search

RUNS = 20
LAB_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_FOLDER = os.path.join(LAB_FOLDER, "configs")
RESULTS_FOLDER = os.path.join(LAB_FOLDER, "results")


def load_config(name):
    path = os.path.join(CONFIGS_FOLDER, name + ".json")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_runs(name, rows):
    path = os.path.join(RESULTS_FOLDER, "runs_" + name + ".csv")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["config", "seed", "best_f", "best_x", "n_evals"])
        writer.writerows(rows)


def write_history(name, first_seed, histories):
    path = os.path.join(RESULTS_FOLDER, "traj_" + name + ".csv")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["config", "seed", "generation", "best_f"])
        for run in range(len(histories)):
            for generation in range(len(histories[run])):
                writer.writerow([
                    name,
                    first_seed + run,
                    generation,
                    histories[run][generation],
                ])


def run_series(config, method):
    if method == "random":
        name = "random"
    else:
        name = config["name"]

    rows = []
    values = []
    histories = []
    first_seed = config["seed_start"]
    for seed in range(first_seed, first_seed + RUNS):
        if method == "random":
            result = run_random_search(config, seed)
        else:
            result = run_ga(config, seed)

        coordinates = []
        for value in result["best_x"]:
            coordinates.append(str(value))
        vector_text = " ".join(coordinates)

        rows.append([name, seed, result["best_f"], vector_text, result["n_evals"]])
        values.append(result["best_f"])
        histories.append(result["history"])
        print(f"{name}, seed {seed}: {result['best_f']}")

    write_runs(name, rows)
    if method == "ga":
        write_history(name, first_seed, histories)
    return values, histories


def write_summary(names, series):
    path = os.path.join(RESULTS_FOLDER, "summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["config", "n_runs", "best", "mean", "median", "std", "worst"])
        for i in range(len(series)):
            values = series[i]
            writer.writerow([
                names[i],
                len(values),
                min(values),
                statistics.mean(values),
                statistics.median(values),
                statistics.stdev(values),
                max(values),
            ])


def plot_convergence(histories):
    generations = list(range(len(histories[0])))
    lowest = []
    average = []
    highest = []
    for generation in generations:
        values = []
        for history in histories:
            values.append(history[generation])
        lowest.append(min(values))
        average.append(statistics.mean(values))
        highest.append(max(values))

    plt.figure()
    plt.fill_between(generations, lowest, highest, color="lightgray", label="минимум–максимум")
    plt.plot(generations, average, label="среднее")
    plt.plot(generations, lowest, label="минимум")
    plt.yscale("log")
    plt.xlabel("Поколение")
    plt.ylabel("Лучшее значение функции")
    plt.title(f"Сходимость базового ГА: {RUNS} запусков")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "convergence_base.png"))
    plt.close()


def plot_comparison(names, series):
    plt.figure()
    plt.boxplot(series)
    plt.xticks(range(1, len(names) + 1), names)
    plt.yscale("log")
    plt.ylabel("Лучшее значение функции за запуск")
    plt.title("Сравнение методов")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "comparison_box.png"))
    plt.close()


def main():
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    base = load_config("base")
    large = load_config("sigma_large")
    base_values, base_histories = run_series(base, "ga")
    large_values, large_histories = run_series(large, "ga")
    random_values, random_histories = run_series(base, "random")

    names = ["ГА σ=0.3", "ГА σ=1.5", "случайный поиск"]
    series = [base_values, large_values, random_values]
    write_summary(names, series)
    plot_convergence(base_histories)
    plot_comparison(names, series)
    print("Таблицы и графики сохранены в папку results.")


if __name__ == "__main__":
    main()
