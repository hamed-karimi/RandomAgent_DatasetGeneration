import numpy as np
import torch
from torch.nn import ReLU
import random
from copy import deepcopy
from itertools import product


class Agent:
    def __init__(self, h, w, n, agent_type, prob_init_needs_equal, predefined_location, preassigned_needs):  # n: number of needs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = h
        self.width = w
        self.AGENT_TYPE_OPTIONS = ["Regular", "RandomNeeds", "NearestObject"]
        self.agent_type = agent_type
        self.location = self.initial_location(predefined_location)
        self.num_need = n
        self.range_of_need = [-12, 12]
        self.prob_init_needs_equal = prob_init_needs_equal
        self.need = self.set_need(preassigned_needs)
        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.lambda_need = 1  # How much the need increases after each action
        self.lambda_satisfaction = 3
        self.relu = ReLU()
        total_need_functions = {'ReLU': self.relu, 'PolyReLU': self.poly_relu}
        self.rho_function = total_need_functions['ReLU']
        self.total_need = self.get_total_need()
        possible_h_w = [list(range(h)), list(range(w))]
        self.all_locations = torch.from_numpy(np.array([element for element in product(*possible_h_w)]))
        self.assert_agent_type()

    def assert_agent_type(self):
        assert self.agent_type in self.AGENT_TYPE_OPTIONS, f"'{self.agent_type}' is not in {self.AGENT_TYPE_OPTIONS}"

    def poly_relu(self, x, p=2):
        return self.relu(x) ** p

    def set_need(self, preassigned_needs=None):
        if any(preassigned_needs):
            need = torch.tensor(preassigned_needs)
        else:
            p = random.uniform(0, 1)
            if p <= self.prob_init_needs_equal:
                need = torch.rand((1, self.num_need))
                need[0, 1:] = need[0, 0]
            else:
                need = torch.rand((1, self.num_need))
            need = (self.range_of_need[1] - self.range_of_need[0]) * need + self.range_of_need[0]
        return need

    def initial_location(self, predefined_location): # predefined_location is a list
        if len(predefined_location[0]) > 0:
            return torch.tensor(predefined_location)
        return torch.from_numpy(np.asarray((np.random.randint(self.height), np.random.randint(self.width)))).unsqueeze(0)

    def update_need_after_step(self):
        if self.agent_type in ['Regular', 'NearestObject']:
            for i in range(self.num_need):
                self.need[0, i] += self.lambda_need
        else:
            self.need = self.set_need(preassigned_needs=[[]])

    def update_need_after_reward(self, reward):
        self.need = self.need - reward
        for i in range(self.num_need):
            self.need[0, i] = max(self.need[0, i], -12)

    def get_total_need(self):
        total_need = self.rho_function(self.need).sum().squeeze()
        return total_need

    def get_location(self):
        return self.location

    def take_action(self, environment, action_id):
        selected_action = environment.allactions[action_id].squeeze()  # to device
        self.location[0, :] += selected_action
        self.update_need_after_step()

        environment.update_agent_location_on_map(self)
        f, _ = environment.get_reward()
        self.update_need_after_reward(f)
        at_total_need = self.get_total_need()
        self.total_need = deepcopy(at_total_need)