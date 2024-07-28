import csv
import os.path
from multiprocessing import Pool
from typing import Optional
import numpy as np

from banditsim.graph import Graph
from banditsim.models import AnalyzedResults, DynamicEpsilonConfig, SimResults

def process(grid, path):
    for params in grid:
        print(params)
        n_simulations, graph, a, n, e, max_epochs, burn_in, epsilon_changes = params
        pool = Pool()
        results = pool.starmap(
            run_simulation, ((graph, a, n, e, max_epochs, burn_in, epsilon_changes),) * n_simulations)
        pool.close()
        pool.join()
        # for _ in range(n_simulations):
        #     results = run_simulation(graph, a, n, e, m, max_epochs, burn_in)
        pathname, extension = os.path.splitext(path)
        record_data_dump(results, pathname + '_datadump' + extension)
        record_analysis(analyzed_results(results), path)

def run_simulation(graph, a, n, e, max_epochs, burn_in, epsilon_changes: Optional[DynamicEpsilonConfig]):
    g = Graph(a, graph, max_epochs, e, epsilon_changes)
    g.run_simulation(n, burn_in)
    e_change_n_rounds = epsilon_changes.change_after_n_rounds if epsilon_changes else None
    epsilon_d = epsilon_changes.epsilon_d if epsilon_changes else None
    return SimResults(graph, a, max_epochs, n, e, burn_in, e_change_n_rounds,
                      epsilon_d, g.epoch, g.av_utility)

def record_data_dump(simresults: list[SimResults], path):
    file_exists = os.path.isfile(path)
    with open(path, mode = 'a') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([header for header in simresults[0]._asdict().keys()])
        for simresult in simresults:
            writer.writerow([result_val for result_val in simresult])

def record_analysis(analyzed_results: AnalyzedResults, path):
    file_exists = os.path.isfile(path)
    with open(path, mode = 'a') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([header for header in analyzed_results._asdict().keys()])
        writer.writerow([result_val for result_val in analyzed_results])

def analyzed_results(simresults: list[SimResults]):
    av_utility = round(np.mean([res.av_utility for res in simresults]), 3)
    sim = simresults[0] # grab metadata/params
    return AnalyzedResults(sim.graph_shape, sim.agents, sim.max_epochs, sim.trials, sim.epsilon,
                           sim.burn_in, sim.e_change_n_rounds, sim.epsilon_d, av_utility)
