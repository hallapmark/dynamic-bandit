
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
    # sim_average_expectation = None
    sim_total_utility = 0
    sim_average_utility = None

    def record_sim_end_metrics(self, g, trials):
        self.record_proportion_correct_action(g)
        self.record_sim_average_utility(g, trials)
        # print(f'n_rounds_all_took_A: {self.n_rounds_all_took_A}')
        # print(f'n_rounds_all_took_B: {self.n_rounds_all_took_B}')
        # print(f'n_rounds_supermajority_took_A: {self.n_rounds_supermajority_took_A}')
        # print(f'n_rounds_supermajority_took_B: {self.n_rounds_supermajority_took_B}')
        # print(f'n_rounds_mixed_actions: {self.n_rounds_supermajority_took_B}')

    def record_round_metrics(self, g):
        g: graph.Graph = g
        self.record_round_correct_actions(g)
        self.record_round_taken_actions(g)
        self.record_round_util(g)
    
        # self.record_round_average_expectation(g)
    
    def record_round_correct_actions(self, g):
        g: graph.Graph = g
        epsilon = g.epsilons[g.epoch]
        b_better = epsilon >= 0
        self.correct_actions.append("B" if b_better else "A")
    
    def record_round_taken_actions(self, g):
        g: graph.Graph = g
        a_list = [a.round_action for a in g.agents]
        self.taken_actions.append(a_list)
        self.record_network_action_state(g, a_list)

    # def record_round_average_expectation(self, g):
    #     g: graph.Graph = g
    #     self.average_expectations.append(np.average([a.expectation_B for a in g.agents]))

    def record_round_util(self, g):
        g: graph.Graph = g
        # each win/success k of the n pulls, has a util of "1"
        # We sum up the utils from action A and B to get the total util agents managed to pull
        self.sim_total_utility += sum(
            [a.action_A_data[-1].k for a in g.agents if a.action_A_data and a.round_action == "A"])
        self.sim_total_utility += sum(
            [a.action_B_data[-1].k for a in g.agents if a.action_B_data and a.round_action == "B"])

    def record_network_action_state(self, g, a_list: list):
        g: graph.Graph = g
        if not a_list:
            return
        
        i = 1
        if all(a == "A" for a in a_list):
            #print(f"Whole network took action 'A' in round {g.epoch}")
            self.n_rounds_all_took_A += 1
        elif all(a == "B" for a in a_list):
            #print(f"Whole network took action 'B' in round {g.epoch}")
            self.n_rounds_all_took_B += 1
        elif a_list.count("A") >= len(a_list) - i:
            #print(f"Supermajority took action 'A' in round {g.epoch}")
            self.n_rounds_supermajority_took_A += 1
        elif a_list.count("B") >= len(a_list) - i:
            #print(f"Supermajority took action 'B' in round {g.epoch}")
            self.n_rounds_supermajority_took_B += 1
        else:
            #print(f"Network took mixed actions in round {g.epoch}")
            self.n_rounds_mixed_actions += 1

    # def update_final_round_when_action_changed(self, g):
    #     g: graph.Graph = g
    #     """ The last round that at least someone moved from action A to B
    #     or vice versa. If this number is low, that means the network
    #     got stuck early on one of the actions."""
        
    def record_proportion_correct_action(self, g):
        # TODO: Verify this
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
    
    # def record_sim_average_expectation(self):
    #     self.sim_average_expectation = np.average(self.average_expectations)
    
    def record_sim_average_utility(self, g, trials):
        g: graph.Graph = g
        # Av utility per round per agent per trial
        self.sim_average_utility = (self.sim_total_utility / len(g.agents)) / g.epoch / trials
        