from banditsim.models import GraphShape
from banditsim.sim import process
import timeit
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    s = 10000
    max_epochs = 4000
    ## Zollman 2007 but with expectation updating. 
    grid = [(s, GraphShape.COMPLETE, a, 1000, .001, max_epochs, burn_in, None) 
            for a in range(9, 11) # 4, 12
            for burn_in in range(0, 2)]
    grid += [(s, GraphShape.CYCLE, a, 1000, .001, max_epochs, burn_in, None) 
             for a in range(9, 11)
             for burn_in in range(0, 2)]
    
    # Larger epsilon (Rosenstock et al. 2017 but again with expectation updating)
    # grid += [(s, GraphShape.COMPLETE, a, 1000, .005, max_rounds, burn_in, None) 
    #          for a in range(9, 11) # 4, 12
    #          for burn_in in range(0, 2)]
    # grid += [(s, GraphShape.CYCLE, a, 1000, .005, max_rounds, burn_in, None) 
    #          for a in range(9, 11)
    #          for burn_in in range(0, 2)]

    tic = timeit.default_timer()
    process(grid, 'results/zollman_expectation.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
