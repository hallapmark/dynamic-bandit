import timeit
from multiprocessing import freeze_support

from banditsim.models import GraphShape
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    n_simulations = 200
    max_epochs = 4000

    grid = [(n_simulations, GraphShape.COMPLETE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon,
             window_s) 
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0.1,)
                                                                for window_s in (None,) # 50, 100
                                                                ]
    
    grid += [(n_simulations, GraphShape.CYCLE, a, 50, sine_amp, sine_period, max_epochs, burn_in, epsilon, 
             window_s)
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for epsilon in (0.1,)
                                                                for window_s in (None,)
                                                                ]

    tic = timeit.default_timer()
    process(grid, 'results/test.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
