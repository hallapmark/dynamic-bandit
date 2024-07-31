
from dataclasses import dataclass

import numpy as np
from . import graph

@dataclass
class SimMetrics:

    ## Per round tallies
    correct_actions = [] # E.g. ["A", "A", "B" ...] for correct action in round 0, 1, 2 ...
    taken_actions = [] # [[]] taken_actions[i] is a list of actual actions taken by agents in round i
    round_average_expectations = [] # Average expectation of network in round i
    
    ## Sim-end summaries
    proportion_correct_action = None
    
    def record_round_metrics(self, g):
        self.record_round_correct_actions(g)
        self.record_round_taken_actions(g)
    
    def record_round_correct_actions(self, g):
        g: graph.Graph = g
        epsilon = g.epsilons[g.epoch]
        b_better = epsilon >= 0
        self.correct_actions.append("B" if b_better else "A")
    
    def record_round_taken_actions(self, g):
        g: graph.Graph = g
        self.taken_actions.append([a.round_action for a in g.agents])

    def record_round_average_expectation(self, g):
        g: graph.Graph = g
        self.round_average_expectations.append(np.average([a.expectation_B for a in g.agents]))

    def record_sim_end_metrics(self, g):
        self.proportion_correct_action = self.record_proportion_correct_action(g)

    def update_final_round_when_action_changed(self, g):
        g: graph.Graph = g 
        """ The last round that at least someone moved from action A to B
        or vice versa. If this number is low, that means the network
        got stuck early on one of the actions."""
        
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
        return np.average(prop_list)
        