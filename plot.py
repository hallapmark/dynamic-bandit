import numpy as np

import matplotlib.pyplot as plot

class PlotSine:
    def __init__(self, max_epochs, sine_epsilons: np.ndarray, agent_expectations):
        self.epochs = np.arange(0, max_epochs, 1)
        self.p = [0.5 + e for e in sine_epsilons]
        self.agent_expectations = agent_expectations

    def makePlot(self):
        plot.plot(self.epochs, self.p, label = "p")
        plot.plot(self.epochs, self.agent_expectations, label = "expectation")
        plot.title('Sine epsilons and agent expectations')
        plot.xlabel('Time')
        plot.ylabel('Amplitude = sin(time)')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.legend() 
        plot.show()
    