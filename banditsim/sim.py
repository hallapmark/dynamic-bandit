import csv
import os.path
from multiprocessing import Pool
from banditsim.graph import Graph
from banditsim.models import SimResults

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
        record_data_dump(results, path)

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
