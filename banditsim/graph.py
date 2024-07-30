import numpy as np

from banditsim.agent import Agent
from banditsim.models import GraphShape

class Graph:
    def __init__(self, a: int, shape: GraphShape, max_epochs: int, max_epsilon: float, epsilon_sine_period: float):
        np.random.seed()
        self.agents = [Agent() for _ in range(a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self.epoch = 0
        self.max_epochs = max_epochs
        self.max_epsilon = max_epsilon
        self.epsilon_sine_period = epsilon_sine_period

        if shape == GraphShape.CYCLE:
            for i in range(a):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % a] ]
        elif shape == GraphShape.COMPLETE:
            for i in range(a):
                self.graph[self.agents[i]] = self.agents
    
    def __str__(self):
        return "\n" + "\n".join([str(a) for a in self.agents])

    def run_simulation(self, n: int, burn_in: int):
        self.run_burn_in(n, burn_in, .5 + self.max_epsilon)
        t = np.arange(0, self.max_epochs, 1)
        self.epsilons = self.sine_epsilons(self.max_epsilon, t, self.epsilon_sine_period)

        ## TODO: Verify that this runs *exactly* the number of times we want
        while self.epoch < self.max_epochs:
            self.run_experiments(n, self.epoch)
            self.expectation_update_agents()
            self.epoch += 1

        # each win/success k of the n pulls, has a util of "1"
        # We sum up the utils from action A and B to get the total util agents managed to pull
        tot_utility = sum([a.action_A_data.k for a in self.agents] + [a.action_B_data.k for a in self.agents])
        self.av_utility = (tot_utility / len(self.agents)) / self.epoch / n
        # Av utility per round per agent per trial

    def run_burn_in(self, n, burn_in, p):
        burn_in_rounds = 0
        while burn_in_rounds < burn_in:
            burn_in_rounds += 1
            for a in self.agents:
                a.burn_in(n, p)

    def run_experiments(self, n, epoch):
        e = self.epsilons[epoch]
        p = 0.5 + e
        for a in self.agents:
            a.decide_experiment(n, p)
        
    def expectation_update_agents(self):
        for a in self.agents:
            total_k, total_n = 0, 0
            for neighbor in self.graph[a]:
                total_k += neighbor.action_B_data.k
                total_n += neighbor.action_B_data.n
            total_k += a.private_B_data.k
            total_n += a.private_B_data.n
            if total_n > 0:
                a.expectation_B_update(total_k, total_n)
    
    def sine_epsilons(self, max_epsilon: float, t: np.ndarray, period: int):
        """ Returns a numpy array of epsilons (floats) shaped like a sine wave
        fluctuating between max_epsilon and -max_epsilon in amplitude."""
        b = 2 * np.pi  / period
        return max_epsilon * np.sin(b * (t + period / 4))
        