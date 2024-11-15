import timeit
from multiprocessing import freeze_support

from banditsim.models import AdmitteeType, GraphShape
from banditsim.sim import process

if __name__ == '__main__':
    freeze_support()
    n_simulations = 200
    max_epochs = 4000

    grid = [(n_simulations, GraphShape.COMPLETE, a, 50, sine_amp, sine_period, max_epochs, burn_in, 
             window_s, lifecycle, admitteetype)
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for window_s in (None,) # 50, 100
                                                                for lifecycle in (False,)
                                                                for admitteetype in (None,)]
    
    grid += [(n_simulations, GraphShape.CYCLE, a, 50, sine_amp, sine_period, max_epochs, burn_in, 
             window_s, lifecycle, admitteetype)
                                                                for a in (10,) # 4, 12
                                                                for sine_amp in (.1,)
                                                                for sine_period in (1000,)
                                                                for burn_in in (1,)
                                                                for window_s in (None,)
                                                                for lifecycle in (False,)
                                                                for admitteetype in (None,)]

    tic = timeit.default_timer()
    process(grid, 'results/test.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
