from banditsim.models import DynamicEpsilonConfig, GraphShape
from banditsim.sim import process
import timeit
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    s = 10000
    max_rounds = 8000
    ## Zollman 2007 but with expectation updating. 
    # grid = [(s, GraphShape.COMPLETE, a, 1000, .001, None, max_rounds, burn_in, None) for a in range(9, 12) # 4, 12
    #                                                                             for burn_in in range(0, 3)]
    # grid += [(s, GraphShape.CYCLE, a, 1000, .001, None, max_rounds, burn_in, None) for a in range(9, 12)
    # 
    #                                                                             for burn_in in range(0, 3)]
    
    # Larger epsilon
    # grid = [(s, GraphShape.COMPLETE, a, 1000, .005, None, max_rounds, burn_in, None) for a in range(9, 12) # 4, 12
    #                                                                             for burn_in in range(0, 3)]
    # grid += [(s, GraphShape.CYCLE, a, 1000, .005, None, max_rounds, burn_in, None) for a in range(9, 12)
    #                                                                             for burn_in in range(0, 3)]
    ## Dynamic bandit:
    grid = [(s, GraphShape.COMPLETE, a, 50, .005, None, max_rounds, burn_in, DynamicEpsilonConfig(80, 0.0001))
                                                                                for a in range(6, 8) # 4, 12
                                                                                for burn_in in range(0, 2)]
    
    grid += [(s, GraphShape.CYCLE, a, 50, .005, None, max_rounds, burn_in, DynamicEpsilonConfig(80, 0.0001))
                                                                                for a in range(6, 8) # 4, 12
                                                                                for burn_in in range(0, 2)]


    tic = timeit.default_timer()
    process(grid, 'results/zollman_expectation_epsilon_changes_n50_e_start_005_change_80_test.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
