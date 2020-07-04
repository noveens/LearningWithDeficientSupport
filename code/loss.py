import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import *

class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(CustomLoss, self).__init__()
        self.hyper_params = hyper_params
        self.k = float(hyper_params['k'])

    def forward(self, output, y, action, delta, prop, all_prop, regressed_rewards, return_mean = True):
        risk = -1.0 * delta

        if self.hyper_params['estimator_type'] == 'IPS':
            loss  = risk * (output[range(action.size(0)), action] / prop)

        elif self.hyper_params['estimator_type'] == 'RegressionExtrapolation': 
            unsupp_actions = (all_prop == 0).float()
            rewards_for_unsupp_actions = regressed_rewards * output * unsupp_actions
            
            loss  = risk * (output[range(action.size(0)), action] / prop)
            loss -= torch.sum(rewards_for_unsupp_actions, -1)

        elif self.hyper_params['estimator_type'] == 'PolicyRestriction':
            loss = (risk + self.k) * (output[range(action.size(0)), action] / prop)
            loss -= self.k # Only translation, but still to be precise. Ignoring epsilon terms
            
        if return_mean == True: loss = torch.mean(loss)
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(MSELoss, self).__init__()
        self.hyper_params = hyper_params

    def forward(self, delta_hat, y, action, delta, prop, all_prop, regressed_rewards, return_mean = True):
        loss = torch.pow(delta_hat[range(action.size(0)), action] - delta, 2)
        if return_mean == True: loss = torch.mean(loss)
        return loss
