import csv
import os.path
from multiprocessing import Pool, cpu_count
import numpy as np

from banditsim.graph import Graph
from banditsim.models import AnalyzedResults, SimConfig, SimResults
from plot_graphs import PlotSine

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
    record_analysis(analyzed_results(results), path)

def run_simulation(config: SimConfig):
    g = Graph(config)
    g.run_simulation(config.n, config.burn_in)
    # plotsine = PlotSine(g.config.max_epochs, g.sine_deltas) # Uncomment to draw plot
    # plotsine.plot_fig1_AB_ob_chance_of_payoff() # Currently plot can only be drawn if multiprocessing is disabled above
    # plotsine.plot_expectation_vs_ob_chance_of_payoff(g.metrics.average_expectations, config.epsilon)
    return SimResults(graph_shape=config.graph_shape, agents=config.a, max_epochs=config.max_epochs, trials=config.n, 
                      sine_amp=config.sine_amp, sine_period=config.sine_period, burn_in=config.burn_in, 
                      epsilon=config.epsilon, window_s=config.window_s, lifecycle=False, epochs=g.epoch, 
                      av_utility=g.metrics.sim_average_utility)

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
    av_utility = round(np.mean([res.av_utility for res in simresults]), 7)
    sim = simresults[0] # grab metadata/params
    return AnalyzedResults(sim.graph_shape, sim.agents, sim.max_epochs, sim.trials, sim.sine_amp,
                           sim.sine_period, sim.burn_in, sim.epsilon, sim.window_s, sim.lifecycle, 
                           av_utility)
