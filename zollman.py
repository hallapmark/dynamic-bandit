from banditsim.models import GraphShape, SimParams
from banditsim.sim import process
import timeit
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    s = 10000
    grid = [SimParams(s, GraphShape.COMPLETE, a, 1000, .001, None, 4000) for a in range(4, 12)]
    grid += [SimParams(s, GraphShape.CYCLE, a, 1000, .001, None, 4000) for a in range(4, 12)]

    tic = timeit.default_timer()
    process(grid, 'results/zollman.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
