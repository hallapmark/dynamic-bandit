from typing import Optional
from banditsim.agent import Agent
import numpy as np

from banditsim.models import GraphShape, ResultType

class ZollmanGraph:
    def __init__(self, a: int, shape: GraphShape, max_epochs: int, epsilon: float):
        np.random.seed()
        self.agents = [Agent() for i in range(a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self.epoch = 0
        self.max_epochs = max_epochs
        self.epsilon = epsilon

        if shape == GraphShape.CYCLE:
            for i in range(a):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % a] ]
        elif shape == GraphShape.COMPLETE:
            for i in range(a):
                self.graph[self.agents[i]] = self.agents
    
    def __str__(self):
        return "\n" + "\n".join([str(a) for a in self.agents])

    def run_simulation(self, n: int, burn_in: int):
        burn_in_rounds = 0
        while burn_in_rounds < burn_in:
            burn_in_rounds += 1
            self.run_burn_in(n, self.epsilon)
            
        ## TODO: Verify that this runs *exactly* the number of times we want
        while(self.undecided() and self.epoch < self.max_epochs):
            self.epoch += 1
            self.run_experiments(n, self.epsilon)
            self.expectation_update_agents()
        self.sim_wrapup()

    def sim_wrapup(self):
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

    def run_experiments(self, n, epsilon):
        for a in self.agents:
            a.decide_experiment(n, epsilon)

    def run_burn_in(self, n, epsilon):
        for a in self.agents:
            a.burn_in(n, epsilon)

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

    def undecided(self):
        expectations = np.array([a.expectation_B for a in self.agents])
        #if all credences are .5 or less, then (or) is true. Then returns false. 
        # i.e. the network is not undecided, and the simulation stops 
        return not (all(expectations <= .5) or all(expectations > .99)) 
        # TODO: > .99 doesn't really make sense with expectation updating.
        # But I also don't see any harm in this, we'll just run until max rounds.
        # But this would need fixing if we ever care about the *speed* with which a 
        # network reaches consensus on B. Then, we'd want to detect when we can reasonably
        # say that a network has reached consensus on B.
        