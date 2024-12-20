import csv
import os.path
from multiprocessing import Pool, cpu_count
import numpy as np

from banditsim.graph import Graph
from banditsim.models import AnalyzedResults, SimConfig, SimResults

def process(n_simulations: int, configs: list[SimConfig], path: str, multiprocessing: bool):
    for config in configs:
        process_config(n_simulations, config, path, multiprocessing)

def process_config(n_simulations: int, config: SimConfig, path: str, multiprocessing: bool):
    print(config)
    if multiprocessing:
        pool = Pool(processes=max(cpu_count() - 1, 1)) # Leave one core for Reddit
        results = pool.map(
            run_simulation, (config,) * n_simulations) # n simulations per given config
        pool.close()
        pool.join()
    else:
        results: list[SimResults] = []
        for _ in range(n_simulations):
            results.append(run_simulation(config))
            # breakpoint()
            # break
    pathname, extension = os.path.splitext(path)
    record_data_dump(results, pathname + '_datadump' + extension)
    record_analysis(analyzed_results(results, config), path)
    
def run_simulation(config: SimConfig):
    g = Graph(config)
    g.run_simulation()
    return SimResults(*config, g.epoch, g.metrics.sim_average_utility)

def record_data_dump(simresults: list[SimResults], path):
    file_exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode = 'a') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([header for header in simresults[0]._asdict().keys()])
        for simresult in simresults:
            writer.writerow([result_val for result_val in simresult])

def record_analysis(analyzed_results: AnalyzedResults, path):
    file_exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode = 'a') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([header for header in analyzed_results._asdict().keys()])
        writer.writerow([result_val for result_val in analyzed_results])

def analyzed_results(simresults: list[SimResults], config: SimConfig):
    av_utility = round(np.mean([sim.av_utility for sim in simresults]), 4)
    return AnalyzedResults(len(simresults), *config, False, av_utility)
