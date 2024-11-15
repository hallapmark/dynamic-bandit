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
                 sine_amp: float, 
                 sine_period: float):
        np.random.seed()
        self.agents = [Agent() for _ in range(a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self._epoch = 0
        
        ## Config
        self.max_epochs = max_epochs
        self.sine_amp = sine_amp
        t = np.arange(0, self.max_epochs, 1)
        self.sine_deltas = self.build_sine_deltas(sine_amp, t, sine_period)

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

    def run_simulation(self, n: int, burn_in: int, window_s: Optional[int]):
        self.run_burn_in(n, burn_in, .5 + self.sine_amp)

        ## TODO: Verify that this runs *exactly* the number of times we want
        while self.epoch < self.max_epochs:
            self._play_round(n, window_s)
            
        self.metrics.record_sim_end_metrics(self, n)

    def _play_round(self, n: int, window_s: Optional[int]):
        # if self.epoch % 1000 == 0:
        #     print(f"A sim reached {self.epoch}")
        self._standard_round_actions(n, window_s)
    
    def _standard_round_actions(self, n: int, window_s: Optional[int]):
        self.run_experiments(n, self.epoch)
        for a in self.agents:
            self.update_expectation(a, window_s)
        self.epoch += 1

    def run_burn_in(self, n, burn_in, p):
        burn_in_round = 0
        while burn_in_round < burn_in:
            burn_in_round += 1
            for a in self.agents:
                a.burn_in(n, p)

    def run_experiments(self, n, epoch):
        d = self.sine_deltas[epoch]
        p = 0.5 + d
        for a in self.agents:
            a.decide_experiment(n, p)
    
    def update_expectation(self, a: Agent, window_s: Optional[int]):
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
    
    def build_sine_deltas(self, sine_amp: float, t: np.ndarray, period: int):
        """ Returns a numpy array of deltas (floats) shaped like a sine wave
        fluctuating between sine_amp and -sine_amp."""
        b = 2 * np.pi  / period
        return sine_amp * np.sin(b * (t + period / 4))

class LifecycleGraph(Graph):
    def __init__(self, 
                 a: int, 
                 shape: GraphShape, 
                 max_epochs: int, 
                 sine_amp: float, 
                 sine_period: float,
                 admittee_type: AdmitteeType):
        super().__init__(a, shape, max_epochs, sine_amp, sine_period)
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
