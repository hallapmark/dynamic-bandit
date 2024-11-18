from typing import Optional
import numpy as np

from . import metrics
from banditsim.agent import *
from banditsim.models import GraphShape, SimParams

class Graph:
    def __init__(self, 
                 params: SimParams):
        np.random.seed()
        keep_round_records = params.window_s is not None
        self.agents = [Agent(keep_round_records) for _ in range(params.a)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self._epoch = 0
        
        ## Config
        self.params = params
        t = np.arange(0, params.max_epochs, 1)
        self.sine_deltas = self.build_sine_deltas(params.sine_amp, t, params.sine_period)

        ## Outcome
        self.metrics = metrics.SimMetrics()

        ## Structure the network
        n_agents = len(self.agents)
        if params.graph_shape == GraphShape.CYCLE:
            for i in range(n_agents):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % params.a] ]
        elif params.graph_shape == GraphShape.COMPLETE:
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

    def run_simulation(self, n: int, burn_in: int):
        self.run_burn_in(n, burn_in, .5 + self.params.sine_amp)

        ## TODO: Verify that this runs *exactly* the number of times we want
        while self.epoch < self.params.max_epochs:
            self._play_round(n, self.params.window_s, self.params.epsilon)
            
        self.metrics.record_sim_end_metrics(self, n)

    def _play_round(self, n: int, window_s: Optional[int], epsilon: float):
        # if self.epoch % 1000 == 0:
        #     print(f"A sim reached {self.epoch}")
        self._standard_round_actions(n, window_s, epsilon)
    
    def _standard_round_actions(self, n: int, window_s: Optional[int], epsilon: float):
        self.run_experiments(n, self.epoch, epsilon)
        for a in self.agents:
            # Note: everyone is their own neighbor as well
            a.update_expectation_on_neighbors([neighbor for neighbor in self.graph[a]], window_s)
        self.epoch += 1

    def run_burn_in(self, n, burn_in, p):
        burn_in_round = 0
        while burn_in_round < burn_in:
            burn_in_round += 1
            for a in self.agents:
                a.burn_in(n, p)

    def run_experiments(self, n, epoch, epsilon):
        d = self.sine_deltas[epoch]
        p = 0.5 + d
        for a in self.agents:
            self.metrics.sim_total_utility += a.experiment(n, p, epsilon)
    
    def build_sine_deltas(self, sine_amp: float, t: np.ndarray, period: int):
        """ Returns a numpy array of deltas (floats) shaped like a sine wave
        fluctuating between sine_amp and -sine_amp."""
        b = 2 * np.pi  / period
        return sine_amp * np.sin(b * (t + period / 4))
