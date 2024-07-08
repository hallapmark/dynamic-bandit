from numpy import random
from typing import Optional, NamedTuple


class ExperimentData:
    k: int = 0
    n: int = 0

class Agent:
    def __init__(self):
        self.expectation_B = random.uniform(0, 1)
        self.action_A_data = ExperimentData()
        self.action_B_data = ExperimentData()
        self.private_B_data = ExperimentData()

    def __str__(self):
        return f"expectation = {round(self.expectation_B, 2)}, k = {self.action_B_data.k}, n = {self.action_B_data.n}"
    
    def decide_experiment(self, n, epsilon):
        if self.expectation_B > .5:
            self.experiment_B(n, epsilon)
        else:
            self.experiment_A(n)

    def experiment_A(self, n):
        self.action_A_data.k += random.binomial(n, .5)
        self.action_A_data.n += n

    def experiment_B(self, n, epsilon):
        self.action_B_data.k += random.binomial(n, .5 + epsilon)
        self.action_B_data.n += n

    def burn_in(self, n, epsilon):
        self.private_B_data.k += random.binomial(n, .5 + epsilon)
        self.private_B_data.n += n

    def expectation_B_update(self, k, n):
        self.expectation_B = (k + 1) / (n + 2)

    # def jeffrey_update(self, neighbor, epsilon, m):
    #     n = neighbor.n
    #     k = neighbor.k
        
    #     p_E_H  = (0.5 + epsilon)**k * (0.5 - epsilon)**(n - k)         # P(E|H)  = p^k (1-p)^(n-k)
    #     p_E_nH = (0.5 - epsilon)**k * (0.5 + epsilon)**(n - k)         # P(E|~H) = (1-p)^k p^(n-k)
    #     p_E    = self.credence * p_E_H + (1 - self.credence) * p_E_nH  # P(E) = P(E|H) P(E) + P(E|~H) P(~H)
        
    #     p_H_E  = self.credence * p_E_H / p_E                           # P(H|E)  = P(H) P(E|H)  / P(E)
    #     p_H_nE = self.credence * (1 - p_E_H) / (1 - p_E)               # P(H|~E) = P(H) P(~E|H) / P(~E)
        
    #     # q_E = max(1 - abs(self.credence - neighbor.credence) * m * (1 - p_E), 0)  # O&W's Eq. 1 (anti-updating)
    #     q_E = 1 - min(1, abs(self.credence - neighbor.credence) * m) * (1 - p_E)    # O&W's Eq. 2

    #     self.credence = p_H_E * q_E + p_H_nE * (1 - q_E)               # Jeffrey's Rule
                                                                       # P'(H) = P(H|E) P'(E) + P(H|~E) P'(~E)