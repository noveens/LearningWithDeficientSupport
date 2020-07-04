from tqdm import tqdm
import numpy as np

from data import load_data

def iter(train_reader):
    for ind in range(len(train_reader.x)):
        sampled_action  = train_reader.action[ind]
        all_delta       = train_reader.delta[ind]
        pi_o            = train_reader.prop[ind]

        # Overflow errors
        if pi_o[sampled_action] < 0.00001: continue

        yield sampled_action, all_delta, pi_o, ind

def get_k_risky(train_reader, thresh):

    # Create pi
    pi = {}
    for sampled_action, all_delta, pi_o, ind in iter(train_reader):
    
        sorted_prop_actions = np.argsort(pi_o)

        this_pi = [ 0.0 for i in range(len(sorted_prop_actions)) ]

        mass_left = 1.0
        for action in sorted_prop_actions:
            # To maintain same support 
            if pi_o[action] == 0.0:
                this_pi[action] = 0.0 
                continue

            # Greedy algorithm to give max prob to low prop actions
            if thresh * pi_o[action] >= mass_left: 
                this_pi[action] = mass_left
                break
            else: 
                this_pi[action] = thresh * pi_o[action]
                mass_left -= thresh * pi_o[action]

        pi[ind] = this_pi

    # Get IPS estimate of our greedily constructed policy
    # NOTE: IPS is unbiased since our consturcted policy has same support as the evaluation policy
    ips = 0.0; total = 0.0

    for sampled_action, all_delta, pi_o, ind in iter(train_reader):
        ips += float(all_delta[sampled_action]) * (pi[ind][sampled_action] / float(pi_o[sampled_action]))
        total += 1.0

    return ips / total    

def get_all(train_reader):
    return {
        'MinSup_100': get_k_risky(train_reader, thresh = 100),
    }
