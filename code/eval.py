import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from utils import *

def evaluate(model, criterion, reader, hyper_params, eval_estimators = False, test_set = False):    
    metrics = None
    total_batches = 0.0

    model.eval()

    with torch.no_grad():
        for data in reader.iter(test_set = test_set):
            if not test_set:
                x, y, action, delta, prop, all_prop, all_delta, regressed_rewards = data
                unsupp_actions = all_prop == 0.0
            else:
                x, y, all_delta, unsupp_actions = data

            # Getting probability distribution over actions
            output = model(x)
            if hyper_params['prune_unsupported'] in ['testing', 'both']: output[unsupp_actions] = -1e7
            output = F.softmax(output, dim = 1)

            # Computing metrics
            metrics_this_batch = evaluate_this_batch(
                hyper_params, output, data, criterion, eval_estimators, test_set
            )
            
            if metrics is None: metrics = metrics_this_batch
            else: 
                for m in metrics: metrics[m] += metrics_this_batch[m]

            total_batches += 1.0

    for m in metrics: metrics[m] = round(float(metrics[m]) / total_batches, 4)

    return metrics

def evaluate_this_batch(hyper_params, output, data, criterion, eval_estimators, test_set):
    metrics = {}
    x, y, action, delta, prop = None, None, None, None, None
    all_prop, all_delta, regressed_rewards = None, None, None

    if test_set:
        x, y, all_delta, unsupp_actions = data
    else:
        x, y, action, delta, prop, all_prop, all_delta, regressed_rewards = data

    # Measuring the learned policy's utility:
    # 1. Complete stochastic reward (We term it as `Orcale`)
    metrics['Oracle'] = torch.mean(torch.sum(output * FloatTensor(all_delta), -1))

    # 2. Hardmax accuracy
    _, predicted = torch.max(output.data, 1)
    correct = predicted.eq(y.data).sum(); total = y.size(0)
    metrics['Accuracy'] = 100.0 * correct / total

    # Compute other estimators if validating
    if not test_set:
        metrics['Loss'] = criterion(output, y, action, delta, prop, all_prop, regressed_rewards).data
        
        # Measuring the Control Variate (Needs logging policy only for logged action)
        cv_vector = output[range(action.size(0)), action] / prop
        metrics['ControlVariate'] = torch.mean(cv_vector).data
        
        # Measuring the Support Deficiency of the learned policy 
        # (Just evaluating this number needs propensity of logging policy over ALL actions)
        out_of_support_actions = (all_prop == 0.0).float()
        metrics['SupportDeficiency'] = torch.mean(torch.sum(out_of_support_actions * output, -1))
        
        # 2. Estimated utility using various estimators
        if eval_estimators: 
            metrics.update(evaluate_this_batch_estimators(
                output, delta, regressed_rewards, 
                hyper_params, cv_vector
            ))

    return metrics

def evaluate_this_batch_estimators(output, delta, regressed_rewards, hyper_params, cv_vector):
    # Note that any of the below methods DON'T need the logging policy for ALL actions.
    # They just need the propensity, reward of the chosen action!
    metrics = {}

    # IPS
    metrics['IPS'] = float(torch.mean(delta * cv_vector))
    
    # SNIPS
    metrics['SNIPS'] = metrics['IPS'] / float(torch.mean(cv_vector))
    
    # Direct Model
    if regressed_rewards.shape[0] > 0: # Shape is 0 when we are training the regression model
        metrics['DirectModel'] = float(torch.mean(torch.sum(output * regressed_rewards, -1)))
    
    # MinSup
    for k_name in hyper_params['all_ks']:
        k_test = float(hyper_params['all_ks'][k_name])
        metrics[k_name] = k_test + float(torch.mean((delta - k_test) * cv_vector))

    return metrics
