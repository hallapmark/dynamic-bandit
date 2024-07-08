from banditsim.models import GraphShape
from banditsim.sim import process
import timeit
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    s = 10000
    max_rounds = 4000
    grid  = [(s, GraphShape.COMPLETE, a, 1000, .001, None, max_rounds, burn_in) for a in range(4, 12)
                                                                                for burn_in in range(0, 3)]
    grid += [(s, GraphShape.CYCLE, a, 1000, .001, None, max_rounds, burn_in) for a in range(4, 12)
                                                                                for burn_in in range(0, 3)]

    tic = timeit.default_timer()
    process(grid, 'results/zollman_expectation.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
