import sys
import pickle
import numpy as np
from six.moves import cPickle 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '../code')

from model import RegressionModelCifar
from utils import LongTensor, FloatTensor, load_obj, save_obj

def readfile(path):
    x, delta, prop, action = [], [], [], []
    
    data = load_obj(path)
        
    for line in data:    
        x.append(line[0])
        delta.append(line[1:11])
        prop.append(line[11:21])
        action.append(int(line[-1]))

    return np.array(x), np.array(delta), np.array(prop), np.array(action)

pretrained_on_list = list(map(int, sys.argv[1].split(",")))
taos_list = list(map(int, sys.argv[2].split(",")))
num_sample_list = list(map(int, sys.argv[3].split(",")))
clip_threshold_list = list(map(float, sys.argv[4].split(",")))

for pretrained_on in pretrained_on_list:
    for num_sample in num_sample_list:
        for tao in taos_list:
            for clip_threshold in clip_threshold_list:

                print("Imputing dataset for num sample =", num_sample, end = ", ")
                print("Temperature = " + str(tao) + ", Pretrained on =", pretrained_on, end = ", ")
                print("Clip threshold = " + str(clip_threshold))

                # Load already sampled bandit_dataset
                path  = '../data/cifar/bandit_data/' 
                path += 'pretrained_on_' + str(pretrained_on) + '_'
                path += 'tao_' + str(tao) + '_'
                path += 'sampled_' + str(num_sample) + '_'
                path += 'clip_threshold_' + str(clip_threshold)

                x, delta, prop, action = readfile(path + '_train')

                unsupp_x, unsupp_y, unsupp_prop, unsupp_actions = [], [], [], []

                for point_num in tqdm(range(x.shape[0])):
                    unsupported_indices = (prop[point_num] == 0).nonzero()[0]
                    supported_indices = (prop[point_num] != 0).nonzero()[0]
                    total_actions = len(prop[point_num])

                    if len(unsupported_indices) > 0:
                        fake_rewards = [ -12345 for i in range(total_actions) ]

                        fake_prop = []
                        for i in range(total_actions):
                            if i in unsupported_indices: fake_prop.append(1.0 / float(len(unsupported_indices)))
                            else: fake_prop.append(0.0)

                        for _ in range(num_sample):
                            # Sample one item uniformly from unsupported actions
                            sampled_unsupported = unsupported_indices[np.random.randint(len(unsupported_indices))]

                            unsupp_x.append([ x[point_num] ])
                            unsupp_y.append(fake_rewards)
                            unsupp_prop.append(fake_prop)
                            unsupp_actions.append([ sampled_unsupported ])

                unsupp_x = np.array(unsupp_x)
                unsupp_y = np.array(unsupp_y)
                unsupp_prop = np.array(unsupp_prop)
                unsupp_actions = np.array(unsupp_actions)

                # Save as CSV
                if len(unsupp_x) > 0: final_imputed = np.concatenate((unsupp_x, unsupp_y, unsupp_prop, unsupp_actions), axis=1)
                else: final_imputed = np.array([])

                print("number of imputed bandit data points =", len(final_imputed))

                impute_file_path  = '../data/cifar/imputed_data/' 
                impute_file_path += 'pretrained_on_' + str(pretrained_on) + '_'
                impute_file_path += 'tao_' + str(tao) + '_'
                impute_file_path += 'sampled_' + str(num_sample) + '_'
                impute_file_path += 'clip_threshold_' + str(clip_threshold)

                save_obj(final_imputed, impute_file_path)
