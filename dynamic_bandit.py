import timeit
from multiprocessing import freeze_support

from banditsim.models import GraphShape, SimParams
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    n_simulations = 200
    max_epochs = 4000

    grid = [SimParams(GraphShape.COMPLETE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon,
             window_s, slow_updater_multiplier) 
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0,)
                                                                for window_s in (None,) # 50, 100
                                                                for slow_updater_multiplier in (None,)
                                                                ]
    
    grid += [SimParams(GraphShape.CYCLE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon, 
             window_s, slow_updater_multiplier)
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0,)
                                                                for window_s in (None,)
                                                                for slow_updater_multiplier in (None,)
                                                                ]

    tic = timeit.default_timer()
    process(n_simulations, grid, 'results/test.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
