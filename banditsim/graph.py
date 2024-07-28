from banditsim.agent import Agent
import numpy as np
from typing import Optional

from banditsim.models import DynamicEpsilonConfig, GraphShape

class Graph:
    def __init__(self, a: int, shape: GraphShape, max_epochs: int, epsilon: float, epsilon_changes: Optional[DynamicEpsilonConfig] = None):
        np.random.seed()
        self.agents = [Agent() for i in range(a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self.epoch = 0
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.epsilon_changes = epsilon_changes
        if self.epsilon_changes:
            self.rounds_to_e_change = self.epsilon_changes.change_after_n_rounds

        if shape == GraphShape.CYCLE:
            for i in range(a):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % a] ]
        elif shape == GraphShape.COMPLETE:
            for i in range(a):
                self.graph[self.agents[i]] = self.agents
    
    def __str__(self):
        return "\n" + "\n".join([str(a) for a in self.agents])

    def run_simulation(self, n: int, burn_in: int):
        self.run_burn_in(n, burn_in, self.epsilon)

        ## TODO: Verify that this runs *exactly* the number of times we want
        while self.should_continue():
            self.epoch += 1
            self.run_experiments(n, self.epsilon)
            self.expectation_update_agents()
            if self.epsilon_changes:
                self.change_epsilon(self.epsilon_changes)

        # each win/success k of the n pulls, has a util of "1"
        # We sum up the utils from action A and B to get the total util agents managed to pull
        tot_utility = sum([a.action_A_data.k for a in self.agents] + [a.action_B_data.k for a in self.agents])
        self.av_utility = (tot_utility / len(self.agents)) / self.epoch # Av round utility per agent

    def should_continue(self):
        return self.epoch < self.max_epochs

    def run_burn_in(self, n, burn_in, epsilon):
        burn_in_rounds = 0
        while burn_in_rounds < burn_in:
            burn_in_rounds += 1
            for a in self.agents:
                a.burn_in(n, epsilon)

    def run_experiments(self, n, epsilon):
        for a in self.agents:
            a.decide_experiment(n, epsilon)
        
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
    
    def change_epsilon(self, epsilon_changes: DynamicEpsilonConfig):
        self.rounds_to_e_change -= 1
        if self.rounds_to_e_change > 0:
            return
        self.epsilon -= epsilon_changes.epsilon_d
        self.rounds_to_e_change = epsilon_changes.change_after_n_rounds
        