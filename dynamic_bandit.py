import timeit
from multiprocessing import freeze_support

from banditsim.models import GraphShape
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    s = 10000
    max_epochs = 8000

    grid = [(s, GraphShape.COMPLETE, a, 10, .05, sine_period, max_epochs, burn_in)
                                                                for a in range(6, 8) # 4, 12
                                                                for burn_in in range(1, 2)
                                                                for sine_period in (1000,)]
    
    grid += [(s, GraphShape.CYCLE, a, 10, .05, sine_period, max_epochs, burn_in)
                                                                for a in range(6, 8) # 4, 12
                                                                for burn_in in range(1, 2)
                                                                for sine_period in (1000,)]


    tic = timeit.default_timer()
    process(grid, 'results/dynamic_bandit_n50_emax_05_sine_1000.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
