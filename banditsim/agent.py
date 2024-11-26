from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from numpy import random

@dataclass
class ExperimentData:
    k: int = 0
    n: int = 0

class Agent:
    def __init__(self, keep_round_records: bool, trials_per_round: int):
        self.expectation_B = random.uniform(0, 1)
        self.keep_round_records = keep_round_records
        self.trials_per_round = trials_per_round

        # All-time k, n
        self._A_data_total = ExperimentData(0, 0)
        self._B_data_total = ExperimentData(0, 0)
        self._private_B_data = ExperimentData(0, 0)

        # Round-by-round records of B trials
        self._B_round_by_round_k: list[int] = [] # Each element is one round's payoff count k

        self.round_action = "" # "A" or "B"
        self.age = 0

    def __str__(self):
        return (f"expectation = {round(self.expectation_B, 2)}, " 
                f"k = {self._B_data_total.k}, n = {self._B_data_total.n}, "
                f"burn_in k: {self._private_B_data.k}, burn_in n: {self._private_B_data.n}")
    
    def experiment(self, p, epsilon):
        """ Experiment. Parameters:\n
        p: objective probability of payoff from B. (For A, it is always .5)\n
        epsilon: probability that the agent will explore rather than exploit (epsilon-greedy strategy). 
        If 0, then the agent always takes the exploit action (myopic strategy). 

        Returns utility pulled from any source (= k, number of successes this round). 
        """
        self.age += 1 
        if self.expectation_B >= .5 or self.decide_to_explore(epsilon): 
            return self.experiment_B(self.trials_per_round, p)
        else:
            return self.experiment_A(self.trials_per_round)
        
    def decide_to_explore(self, epsilon: float) -> bool:
        return random.choice([True, False], 1, p=[epsilon, 1-epsilon])[0]

    def experiment_A(self, n) -> int:
        """ Experiment with A. Returns k (number of successes this round)"""
        self.round_action = "A"
        k = random.binomial(n, .5)
        self._A_data_total = ExperimentData(self._A_data_total.k + k, self._A_data_total.n + n)
        return k

    def experiment_B(self, n, p) -> int:
        """ Experiment with B. Returns k (number of successes this round)"""
        self.round_action = "B"
        k = random.binomial(n, p)
        self._B_data_total = ExperimentData(self._B_data_total.k + k, self._B_data_total.n + n)
        if self.keep_round_records:
            self._B_round_by_round_k.append(k)
        return k
    
    def burn_in(self, n, p):
        k = random.binomial(n, p)
        self._private_B_data = ExperimentData(self._private_B_data.k + k, self._private_B_data.n + n)

    def update_expectation_on_neighbors(self, neighbors: list[Agent], window_s: Optional[int]):
        total_k, total_n = 0, 0
        for neighbor in neighbors: 
            k, n = neighbor.report_exp_B_data(window_s)
            total_k += k
            total_n += n
        # add burn-in data
        total_k += self._private_B_data.k # Note, for now this is added even for window updaters
        total_n += self._private_B_data.n
        if total_n > 0:
            self.expectation_B_update(total_k, total_n)
        
    def expectation_B_update(self, k, n):
        self.expectation_B = k / n

    def report_exp_B_data(self, window_s: Optional[int]):
        """ Reports all-time experiment B_data or within a window size as requested."""
        if window_s:
            if not self.keep_round_records:
                raise ValueError("Window-sized experiment data requested but round records not kept.")
            windowed_k_list = self._B_round_by_round_k[-window_s:]
            k = sum(windowed_k_list)
            n = self.trials_per_round * len(windowed_k_list)
            return k, n
        else:
            B_data = self._B_data_total
            return B_data.k, B_data.n
