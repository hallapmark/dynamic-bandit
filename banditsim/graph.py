from typing import Optional
import numpy as np

from . import metrics
from banditsim.agent import *
from banditsim.models import GraphShape, SimConfig

class Graph:
    def __init__(self, 
                 config: SimConfig):
        np.random.seed()
        keep_round_records = config.window_s is not None
        self.agents = [Agent(keep_round_records, config.trials) for _ in range(config.agents)]
        self.graph: dict[Agent, list[Agent]] = dict()
        self._epoch = 0
        
        ## Config
        self.config = config
        t = np.arange(0, config.max_epochs, 1)
        self.sine_deltas = self.build_sine_deltas(config.sine_amp, t, config.sine_period)

        ## Outcome
        self.metrics = metrics.SimMetrics()

        ## Structure the network
        n_agents = len(self.agents)
        if config.graph_shape == GraphShape.CYCLE:
            for i in range(n_agents):
                self.graph[self.agents[i]] = [ self.agents[i - 1], self.agents[i], self.agents[(i + 1) % config.agents] ]
        elif config.graph_shape == GraphShape.COMPLETE:
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

    def run_simulation(self):
        self.run_burn_in(self.config.trials, self.config.burn_in, .5 + self.config.sine_amp)

        while self.epoch < self.config.max_epochs:
            self._play_round(self.config.window_s, self.config.epsilon)
            
        self.metrics.record_sim_end_metrics(self, self.config.trials)

    def _play_round(self, window_s: Optional[int], epsilon: float):
        self.run_experiments(self.epoch, epsilon)
        for a in self.agents:
            # Note: everyone is their own neighbor as well
            a.update_expectation_on_neighbors(self.graph[a], window_s)
        self.epoch += 1

    def run_burn_in(self, n, burn_in, p):
        burn_in_round = 0
        while burn_in_round < burn_in:
            burn_in_round += 1
            for a in self.agents:
                a.burn_in(n, p)

    def run_experiments(self, epoch, epsilon):
        d = self.sine_deltas[epoch]
        p = 0.5 + d
        for a in self.agents:
            self.metrics.sim_total_utility += a.experiment(p, epsilon)
    
    def build_sine_deltas(self, sine_amp: float, t: np.ndarray, period: int):
        """ Returns a numpy array of deltas (floats) shaped like a sine wave
        fluctuating between sine_amp and -sine_amp."""
        b = 2 * np.pi  / period
        return sine_amp * np.sin(b * (t + period / 4))
