import csv
import os.path
from multiprocessing import Pool
from statistics import mean

import numpy as np
from banditsim.graph import Graph
from banditsim.models import AnalyzedResults, ResultType, SimResults

def process(grid, path):
    for params in grid:
        print(params)
        n_simulations, graph, a, n, e, m, max_epochs = params
        pool = Pool()
        results = pool.starmap(run_simulation, ((graph, a, n, e, m, max_epochs),) * n_simulations)
        pool.close()
        pool.join()
        # for _ in range(n_simulations):
        #     results = run_simulation(graph, a, n, e, m, max_epochs)
        pathname, extension = os.path.splitext(path)
        record_data_dump(results, pathname + '_datadump' + extension)
        record_analysis(analyzed_results(results), path)

def run_simulation(graph, a, n, e, m, max_epochs):
    g = Graph(a, graph, max_epochs)
    g.run_simulation(n, e, m)
    return SimResults(graph, a, g.epoch, g.av_utility, g.result, n, e, m)

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
    n_true_consensus = len([res for res in simresults if res.result == ResultType.TRUE_CONSENSUS])
    n_false_consensus = len([res for res in simresults if res.result == ResultType.FALSE_CONSENSUS])
    n_indeterminate_sims = len([res for res in simresults if res.result == ResultType.INDETERMINATE])
    prop_true_cons = round(n_true_consensus / len(simresults), 3)
    prop_false_cons = round(n_false_consensus / len(simresults), 3)
    prop_indeterminate = round(n_indeterminate_sims / len(simresults), 3)

    av_utility = np.mean([res.av_utility for res in simresults])

    sim = simresults[0]
    return AnalyzedResults(sim.graph_shape, sim.agents, av_utility, sim.trials, sim.epsilon, sim.mistrust,
                           prop_true_cons, prop_false_cons, prop_indeterminate)
