from dataclasses import dataclass
from numpy import random

@dataclass
class ExperimentData:
    k: int = 0
    n: int = 0

class Agent:
    def __init__(self):
        self.expectation_B = random.uniform(0, 1)
        self.action_A_data: list[ExperimentData] = [] # Each element is one round's experiment data
        self.action_B_data: list[ExperimentData] = [] 
        self.private_B_data: list[ExperimentData] = []
        self.round_action = "" # "A" or "B"

    def __str__(self):
        return f"expectation = {round(self.expectation_B, 2)}, k = {self.action_B_data.k}, n = {self.action_B_data.n}"
    
    def decide_experiment(self, n, p):
        if self.expectation_B >= .5:
            self.experiment_B(n, p)
        else:
            self.experiment_A(n)

    def experiment_A(self, n):
        self.round_action = "A"
        k = random.binomial(n, .5)
        self.action_A_data.append(ExperimentData(k, n))

    def experiment_B(self, n, p):
        self.round_action = "B"
        k = random.binomial(n, p)
        self.action_B_data.append(ExperimentData(k, n))

    def burn_in(self, n, p):
        k = random.binomial(n, p)
        self.private_B_data.append(ExperimentData(k, n))

    def expectation_B_update(self, k, n):
        self.expectation_B = (k + 1) / (n + 2)
