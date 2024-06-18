from bg.sim import process
import timeit
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    s = 10000
    #grid  = [(s, 'complete', a, 1000, .001, None, 1000) for a in range(4, 12)]
    grid = [(s, 'cycle',    a, 1000, .001, None, 5000) for a in range(4, 12)]

    tic = timeit.default_timer()
    process(grid, 'results/zollman.csv')
    toc = timeit.default_timer()

    print("Time: " + str(round(toc - tic, 1)))
