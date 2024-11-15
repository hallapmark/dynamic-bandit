import numpy as np

import matplotlib.pyplot as plot
from scipy.stats import beta

class PlotSine:
    def __init__(self, max_epochs, sine_deltas: np.ndarray):
        self.epochs = np.arange(0, max_epochs, 1)
        self.p = [0.5 + d for d in sine_deltas]

    def plot_fig1_AB_ob_chance_of_payoff(self):
        ax = plot.gca()
        ax.set_ylim([min(self.p), max(self.p)])
        plot.plot(self.epochs, self.p, label = "$P_B$")
        p_a = np.repeat(.5, len(self.epochs))
        plot.plot(self.epochs, p_a, label = "$P_A$")
        plot.title('Chance of payoff in time')
        plot.xlabel('Time (round of play)')
        plot.ylabel('Probability of payoff')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.legend() 
        plot.show()

    def plot_fig2_expectation_vs_ob_chance_of_payoff(self, agent_expectations):
        ax = plot.gca()
        ax.set_ylim([min(self.p+agent_expectations), max(self.p+agent_expectations)])
        plot.plot(self.epochs, self.p, label = "$P_B$ (objective)")
        plot.plot(self.epochs, agent_expectations, label = "Estimate (subjective)")
        plot.title('Chance of payoff and its estimation (myopic agents)')
        plot.xlabel('Time (round of play)')
        plot.ylabel('Chance of payoff')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.legend() 
        plot.show()


class PlotBeta:
    
    # Parameters for the Beta distribution
    alpha = 80
    beta_val = 20

    def pltBeta(self):
        # Create a range of x values from 0 to 1
        x = np.linspace(0, 1, 1000)
        # # Compute the Beta PDF for each x value
        y = beta.pdf(x, self.alpha, self.beta_val)

        # Plot the Beta distribution
        plot.figure(figsize=(8, 6))
        plot.plot(x, y, label=f'Beta({self.alpha}, {self.beta_val})', color='b')
        plot.title(f'Beta Distribution: Beta({self.alpha}, {self.beta_val})', fontsize=14)
        plot.xlabel('x', fontsize=12)
        plot.xticks(np.arange(min(x), max(x), 0.05))
        plot.ylabel('Density', fontsize=12)
        plot.grid(True)
        plot.legend()
        plot.show()

if __name__ == '__main__':
    betaPlt = PlotBeta().pltBeta()
