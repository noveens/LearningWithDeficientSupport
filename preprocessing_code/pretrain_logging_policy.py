import sys
sys.path.insert(0, '../code')

import torch
from main import main
from hyper_params import hyper_params

pretrain_on_list = list(map(int, sys.argv[1].split(",")))
hyper_params['epochs'] = int(sys.argv[2])

hyper_params['weight_decay'] = 1e-2
hyper_params['batch_log_interval'] = 50

for pretrain_on in pretrain_on_list:
	hyper_params['train_limit'] = pretrain_on

	# Will save trained logging policy at this path
	hyper_params['model_file']  = '../logging_policies/' + str(hyper_params['dataset']) + '/'
	hyper_params['model_file'] += str(hyper_params['train_limit']) + '.pt'

	metrics_test = main(hyper_params, pretrain_full_info = True)
