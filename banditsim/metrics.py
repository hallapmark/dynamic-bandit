
from dataclasses import dataclass

import numpy as np
from . import graph

@dataclass
class SimMetrics:

    ## Per round tallies
    correct_actions = []
    # E.g. ["A", "A", "B"...] – action that oughta be taken in round 0, 1, ... to maximize util

    taken_actions = []
    # [[],[]...] taken_actions[i] is a list of actual actions taken by agents in round i

    average_expectations = []
    # average_expectations[i] is the average expectation of network in round i

    n_rounds_all_took_A = 0
    n_rounds_all_took_B = 0
    n_rounds_supermajority_took_A = 0
    # "supermajority" = n-1 (e.g. 10 agents -> at least 9). 
    n_rounds_supermajority_took_B = 0
    n_rounds_mixed_actions = 0
    
    ## Sim-end summaries
    sim_proportion_correct_action = None
    sim_total_utility = 0
    sim_average_utility = None
    
    def record_sim_end_metrics(self, g, trials):
        self.record_proportion_correct_action(g)
        self.record_sim_average_utility(g, trials)

    def record_round_metrics(self, g):
        g: graph.Graph = g
        self.record_round_correct_actions(g)
        self.record_round_taken_actions(g)
        self.record_round_average_expectation(g)

    def record_round_correct_actions(self, g):
        g: graph.Graph = g
        delta = g.sine_deltas[g.epoch]
        b_better = delta >= 0
        self.correct_actions.append("B" if b_better else "A")

    def record_round_taken_actions(self, g):
        g: graph.Graph = g
        a_list = [a.round_action for a in g.agents]
        self.taken_actions.append(a_list)
        self.record_network_action_state(g, a_list)

    def record_round_average_expectation(self, g):
        g: graph.Graph = g
        self.average_expectations.append(np.average([a.expectation_B for a in g.agents]))

    def record_network_action_state(self, g, a_list: list):
        g: graph.Graph = g
        if not a_list:
            return
        
        i = 1
        if all(a == "A" for a in a_list):
            self.n_rounds_all_took_A += 1
        elif all(a == "B" for a in a_list):
            self.n_rounds_all_took_B += 1
        elif a_list.count("A") >= len(a_list) - i:
            self.n_rounds_supermajority_took_A += 1
        elif a_list.count("B") >= len(a_list) - i:
            self.n_rounds_supermajority_took_B += 1
        else:
            self.n_rounds_mixed_actions += 1

    def record_proportion_correct_action(self, g):
        g: graph.Graph = g
        n_agents = len(g.agents)
        prop_list = []
        for i, correct_action in enumerate(self.correct_actions):
            assert n_agents == len(self.taken_actions[i])
            round_n_correct = len(
                [taken_act for taken_act in self.taken_actions[i] if taken_act == correct_action]
                )
            prop_list.append(round_n_correct / len(self.taken_actions[i]))
        self.sim_proportion_correct_action = np.average(prop_list)

    def record_sim_average_utility(self, g, trials):
        g: graph.Graph = g
        # Av utility per round per agent per trial
        self.sim_average_utility = (self.sim_total_utility / len(g.agents)) / g.epoch / trials
        