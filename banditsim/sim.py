import csv
import os.path
from multiprocessing import Pool
import numpy as np

from banditsim.graph import Graph, LifecycleGraph
from banditsim.models import AdmitteeType, AnalyzedResults, SimResults
from plot import PlotSine

def process(grid, path):
    for params in grid:
        print(params)
        n_simulations, graph, a, n, max_epsilon, sine_period, max_epochs, burn_in, B_fans, window_s, lifecycle, admitteetype = params
        pool = Pool()
        results = pool.starmap(
            run_simulation, ((graph, a, n, max_epsilon, sine_period, max_epochs, burn_in, 
                              B_fans, window_s, lifecycle, admitteetype),) * n_simulations)
        pool.close()
        pool.join()
        # for _ in range(n_simulations):
        #     results = run_simulation(graph, a, n, max_epsilon, sine_period, max_epochs, burn_in, B_fans, window_s, lifecycle, admitteetype)
        #     break
        pathname, extension = os.path.splitext(path)
        record_data_dump(results, pathname + '_datadump' + extension)
        record_analysis(analyzed_results(results), path)

def run_simulation(graph, a, n, max_epsilon, sine_period, max_epochs, burn_in, B_fans, window_s, lifecycle, admitteetype):
    if lifecycle:
        g = LifecycleGraph(a, graph, max_epochs, max_epsilon, sine_period, 0, admitteetype)
        g.run_simulation(n, burn_in, window_s)
    else:
        g = Graph(a, graph, max_epochs, max_epsilon, sine_period, B_fans)
        g.run_simulation(n, burn_in, window_s)
    # plotsine = PlotSine(g.max_epochs, g.epsilons, g.metrics.average_expectations) # Uncomment to draw plot
    # plotsine.makePlot() # Currently plot can only be drawn if multiprocessing is disabled above
    return SimResults(graph, a, max_epochs, n, max_epsilon, sine_period, burn_in, B_fans, window_s, lifecycle, admitteetype,
                      g.epoch, g.metrics.sim_average_utility)

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
    return AnalyzedResults(sim.graph_shape, sim.agents, sim.max_epochs, sim.trials, sim.max_epsilon,
                           sim.sine_period, sim.burn_in, sim.B_fans, sim.window_s, sim.lifecycle, 
                           sim.admitteetype, av_utility)
