from banditsim.agent import Agent
import numpy as np
from typing import Optional

from banditsim.models import DynamicEpsilonConfig, GraphShape, ResultType

class DynamicBanditGraph:
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
        if self.epoch > 0:
            self.av_utility = (tot_utility / len(self.agents)) / self.epoch # Av round utility per agent
            expectations = np.array([a.expectation_B for a in self.agents])
            if all(expectations <= .5):
                self.result = ResultType.FALSE_CONSENSUS
            elif all(expectations > .5):
                self.result = ResultType.TRUE_CONSENSUS
            else:
                self.result = ResultType.INDETERMINATE
        else:
            # TODO: Do we calculate util from networks that epistemically fail on round 1 (that never go to action B)?
            # On average, over the long run, av util per agent per round converges to 0.5 for these networks ... 
            # so maybe no point in calculating it, just set it to .5 as a baseline??? Hmm
            self.av_utility = 0.5
            self.result = ResultType.FALSE_CONSENSUS

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


    # def jeffrey_update_agents(self, epsilon, m):
    #     for a in self.agents:
    #         for neighbor in self.graph[a]:
    #             if neighbor == a and a.n > 0:
    #                 a.bayes_update(a.k, a.n, epsilon)
    #             elif neighbor.n > 0:
    #                 a.jeffrey_update(neighbor, epsilon, m)

    def undecided(self):
        expectations = np.array([a.expectation_B for a in self.agents])
        #if all credences are .5 or less, then (or) is true. Then returns false. 
        # i.e. the network is not undecided, and the simulation stops 
        return not (all(expectations <= .5) or all(expectations > .99)) 

    # def polarized(self, m):
    #     credences = np.array([a.credence for a in self.agents])
    #     if all((credences < .5) | (credences > .99)) & any(credences < .5) & any(credences > .99):
    #         min_believer = min(credences[credences > .99])
    #         max_disbeliever = max(credences[credences < .5])
    #         d = min_believer - max_disbeliever
    #         return m * d >= 1
    #     else:
    #         return False