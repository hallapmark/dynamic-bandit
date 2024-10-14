import timeit
from multiprocessing import freeze_support

from banditsim.models import GraphShape
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    s = 10000
    max_epochs = 8000

    grid = [(s, GraphShape.COMPLETE, a, 50, .2, sine_period, max_epochs, burn_in, window_s)
                                                                for a in (20,) # 4, 12
                                                                for burn_in in (1,)
                                                                for sine_period in (1000,)
                                                                for window_s in (None, 100,)]
    
    # grid += [(s, GraphShape.CYCLE, a, 50, .2, sine_period, max_epochs, burn_in, window_s)
    #                                                             for a in (20,) # 4, 12
    #                                                             for burn_in in (1,)
    #                                                             for sine_period in (1000,)
    #                                                             for window_s in (None, 100,)]


    tic = timeit.default_timer()
    process(grid, 'results/window100_and_none.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
