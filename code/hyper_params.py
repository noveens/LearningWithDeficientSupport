hyper_params = {
    'dont_print': False, # True -- no printing to STDOUT

    'dataset': 'cifar',

    # Bandit dataset options
    'pretrained_on': 35000, # Logging policy pretrained on these many full-info points
    'temperature': 4, # Factor controlling support deficiency in logging policy
    'clip_threshold': 0.01, # Threshold on which action prob was set to 0 in logging policy.
    'num_sample': 1, # Number of actions sampled for each context in the original dataset
    'train_limit': int(1000e3), # Max. amount of bandit feedback to train on (Data-points)

    # Optimization options
    'weight_decay': float(1e-6),
    'lr': 0.002,
    'epochs': 200,
    'batch_size': 128,

    # Estimator/Off-Policy Learning Options
    'estimator_type': 'PolicyRestriction', # OPTIONS: ['IPS', 'RegressionExtrapolation', 'PolicyRestriction']
    'k': 0.15, # Needed only if estimator is "PolicyRestriction"
    'validate_using': 'MinSup_100', # OPTIONS: [ 'MinSup_100', 'DirectModel', 'IPS', 'SNIPS', 'Oracle', 'Accuracy']
    'prune_unsupported': 'none', # OPTIONS: ['none', 'training', 'testing', 'both']
    'imputed': 'none', # OPTIONS: ['none', 'negative']
    'dm_features': 32, # Number of features Regression function was trained on
}

common_path  = hyper_params['dataset'] 
common_path += '_estimator_type_' + hyper_params['estimator_type']
if hyper_params['estimator_type'] == 'PolicyRestriction':
    common_path += '_k_' + str(hyper_params['k'])
common_path += '_prune_' + str(hyper_params['prune_unsupported'])
common_path += '_imputed_' + str(hyper_params['imputed']) 
common_path += '_dm_features_' + str(hyper_params['dm_features'])
common_path += '_tao_' + str(hyper_params['temperature'])
common_path += '_pretrained_on_' + str(hyper_params['pretrained_on'])
common_path += '_num_sample_' + str(hyper_params['num_sample'])
common_path += '_train_limit_' + str(hyper_params['train_limit']) 
common_path += '_clip_threshold_' + str(hyper_params['clip_threshold']) 
common_path += '_wd_' + str(hyper_params['weight_decay'])
common_path += '_validate_using_' + str(hyper_params['validate_using'])

hyper_params['tensorboard_path'] = '../tensorboard_logs/' + common_path
hyper_params['log_file'] = '../saved_logs/' + common_path + '.txt'
hyper_params['model_file'] = '../saved_models/' + common_path + '.pt'
