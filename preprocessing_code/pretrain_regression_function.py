import sys
sys.path.insert(0, '../code')

import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from main import main
from hyper_params import hyper_params

def get_common_name(hyper_params):
    file_name  = 'pretrained_on_' + str(hyper_params['pretrained_on']) + '_'
    file_name += 'num_sample_' + str(hyper_params['num_sample']) + '_'
    file_name += 'tao_' + str(hyper_params['temperature']) + '_'
    file_name += 'clip_threshold_' + str(hyper_params['clip_threshold'])  + '_'
    file_name += 'features_' + str(int(hyper_params['dm_features']))

    return file_name

pretrained_on_list = list(map(int, sys.argv[1].split(",")))
taos_list = list(map(int, sys.argv[2].split(",")))
num_sample_list = list(map(int, sys.argv[3].split(",")))
dm_features_list = list(map(int, sys.argv[4].split(",")))
clip_threshold_list = list(map(float, sys.argv[5].split(",")))

for pretrained_on in pretrained_on_list:
    for num_sample in num_sample_list:
        for tao in taos_list:
            for clip_threshold in clip_threshold_list:
                for dm_features in dm_features_list:
                
                    print(
                        "Training regression function for num sample =", num_sample, 
                        "; tao =", tao, "; logging policy pretrained on =", pretrained_on,
                        "; dm features =", dm_features, "/ 32"
                    )
                    hyper_params['num_sample'] = num_sample
                    hyper_params['temperature'] = tao
                    hyper_params['pretrained_on'] = pretrained_on
                    hyper_params['dm_features'] = dm_features
                    hyper_params['clip_threshold'] = clip_threshold
                    hyper_params['lr'] = 0.004
                    hyper_params['weight_decay'] = float(1e-5)
                    hyper_params['epochs'] = 50

                    file_name = get_common_name(hyper_params)
                    hyper_params['log_file']          = '../saved_logs/regressor_' + file_name + '.txt'
                    hyper_params['tensorboard_path']  = '../tensorboard_logs/regression_log_' + file_name
                    hyper_params['model_file']        = '../regression_models/' + str(hyper_params['dataset']) 
                    hyper_params['model_file']       += '/' + file_name + '.pt'

                    metrics_test = main(hyper_params, train_regressor = True)
