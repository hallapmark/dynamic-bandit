from typing import Optional
import numpy as np

from . import metrics
from banditsim.agent import *
from banditsim.models import AdmitteeType, GraphShape

class Graph:
    def __init__(self, 
                 a: int, 
                 shape: GraphShape, 
                 max_epochs: int, 
                 max_epsilon: float, 
                 epsilon_sine_period: float):
        np.random.seed()
        self.agents = [Agent() for _ in range(a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self._epoch = 0
        
        ## Config
        self.max_epochs = max_epochs
        self.max_epsilon = max_epsilon
        t = np.arange(0, self.max_epochs, 1)
        self.epsilons = self.sine_epsilons(max_epsilon, t, epsilon_sine_period)

        ## Outcome
        self.metrics = metrics.SimMetrics()

        ## Structure the network
        n_agents = len(self.agents)
        if shape == GraphShape.CYCLE:
            for i in range(n_agents):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % a] ]
        elif shape == GraphShape.COMPLETE:
            for i in range(n_agents):
                self.graph[self.agents[i]] = self.agents
    
    @property
    def epoch(self):
        return self._epoch
    
    @epoch.setter
    def epoch(self, value):
        self.metrics.record_round_metrics(self)
        self._epoch = value
    
    def __str__(self):
        return "\n" + "\n".join([str(a) for a in self.agents])

    def run_simulation(self, n: int, burn_in: int, window_s: int):
        self.run_burn_in(n, burn_in, .5 + self.max_epsilon)

        ## TODO: Verify that this runs *exactly* the number of times we want
        while self.epoch < self.max_epochs:
            self._play_round(n, window_s)
            
        self.metrics.record_sim_end_metrics(self, n)

    def _play_round(self, n: int, window_s: int):
        # if self.epoch % 1000 == 0:
        #     print(f"A sim reached {self.epoch}")
        self._standard_round_actions(n, window_s)
    
    def _standard_round_actions(self, n: int, window_s: int):
        self.run_experiments(n, self.epoch)
        self.expectation_update_agents(window_s)
        self.epoch += 1

    def run_burn_in(self, n, burn_in, p):
        burn_in_round = 0
        while burn_in_round < burn_in:
            burn_in_round += 1
            for a in self.agents:
                a.burn_in(n, p)

    def run_experiments(self, n, epoch):
        e = self.epsilons[epoch]
        p = 0.5 + e
        for a in self.agents:
            a.decide_experiment(n, p)
        
    def expectation_update_agents(self, window_s: Optional[int]):
        for a in self.agents:
            total_k, total_n = 0, 0
            for neighbor in self.graph[a]: # Note: everyone is their own neighbor as well
                if window_s:
                    B_data = neighbor.action_B_data[-window_s:]
                else: 
                    B_data = neighbor.action_B_data
                k = sum([exp.k for exp in B_data])
                n = sum([exp.n for exp in B_data])
                total_k += k
                total_n += n
            # add burn-in data
            total_k += sum([exp.k for exp in a.private_B_data])
            total_n += sum([exp.n for exp in a.private_B_data])
            if total_n > 0:
                a.expectation_B_update(total_k, total_n)
    
    def sine_epsilons(self, max_epsilon: float, t: np.ndarray, period: int):
        """ Returns a numpy array of epsilons (floats) shaped like a sine wave
        fluctuating between max_epsilon and -max_epsilon in amplitude."""
        b = 2 * np.pi  / period
        return max_epsilon * np.sin(b * (t + period / 4))

class LifecycleGraph(Graph):
    def __init__(self, 
                 a: int, 
                 shape: GraphShape, 
                 max_epochs: int, 
                 max_epsilon: float, 
                 epsilon_sine_period: float,
                 admittee_type: AdmitteeType):
        super().__init__(a, shape, max_epochs, max_epsilon, epsilon_sine_period)
        self.admittee_type = admittee_type
        
    def _play_round(self, n: int, window_s: int):
        self._standard_round_actions(n, window_s)
        if self.epoch % 10 == 0:
            self._retire_someone(self.admittee_type)
        
    def _retire_someone(self, admitteetype: AdmitteeType):
        eligible = [a for a in self.agents if a.age >= 20]
        if not eligible: 
            return
        
        retiree = np.random.choice(eligible)
        if not retiree:
            raise ValueError("We should not reach this. No retiree agent found.")
        existing_av = np.mean([a.expectation_B for a in self.agents]) 
        # TODO: properly this should be average of all agents NOT the retiree. 
        # (But the difference should be negligible)
        retiree.__init__() # re-initialize retiree to new agent
        match admitteetype:
            case AdmitteeType.CONFORMIST:
                retiree.expectation_B = existing_av
                # TODO: Do we want to have burn_in for newcomers???
            case AdmitteeType.NONCONFORMIST:
                retiree.expectation_B = np.random.uniform(0,1)
