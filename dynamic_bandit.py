import timeit
from multiprocessing import freeze_support

from banditsim.models import GraphShape, SimConfig
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    n_simulations = 500
    max_epochs = 10000

    configs = [SimConfig(GraphShape.COMPLETE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon,
             window_s) 
                                                                for a in (5,10) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0,)
                                                                for window_s in (None,) # 50, 100
                                                                ]
    
    configs += [SimConfig(GraphShape.CYCLE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon, 
             window_s)
                                                                for a in (5,10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0,)
                                                                for window_s in (None,)
                                                                ]

    tic = timeit.default_timer()
    process(n_simulations, configs, 'results/results_prelim.csv', True)
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
