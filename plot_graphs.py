import numpy as np

import matplotlib.pyplot as plot

from banditsim.graph import Graph
from banditsim.models import GraphShape, SimConfig



class PlotSine:
    def plot_fig1_AB_ob_chance_of_payoff(self):                                   
        g = Graph(SimConfig(GraphShape.COMPLETE, 5, 50, .1, 1000, 2500, 1, 0, None))
        g.run_simulation()

        epochs = np.arange(0, g.epoch, 1)
        p = [0.5 + d for d in g.sine_deltas]

        ax = plot.gca()
        ax.set_ylim([min(p), max(p)])
        plot.plot(epochs, p, label = "$P_B$")
        p_a = np.repeat(.5, len(epochs))
        plot.plot(epochs, p_a, label = "$P_A$")
        plot.title('Chance of payoff in time')
        plot.xlabel('Time (round of play)')
        plot.ylabel('Probability of payoff')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.legend() 
        plot.show()

    def plot_fig_2_payoff_vs_estimation_myopic(self):
        g = Graph(SimConfig(GraphShape.COMPLETE, 10, 50, .1, 1000, 1500, 1, 0, None))
        g.run_simulation()

        epochs = np.arange(0, g.epoch, 1)
        p = [0.5 + d for d in g.sine_deltas]
        agent_expectations = g.metrics.average_expectations

        ax = plot.gca()
        ax.set_ylim([min(p+agent_expectations), max(p+agent_expectations)])
        plot.plot(epochs, p, label = "$P_B$ (objective)")
        plot.plot(epochs, agent_expectations, label = "Estimate (subjective)")
        plot.title("Chance of payoff and its estimation (myopic agents)")
        plot.xlabel('Time (round of play)')
        plot.ylabel('Chance of payoff')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.legend() 
        plot.show()

if __name__ == '__main__':
    pltSine = PlotSine()
    #pltSine.plot_fig1_AB_ob_chance_of_payoff()
    pltSine.plot_fig_2_payoff_vs_estimation_myopic()
