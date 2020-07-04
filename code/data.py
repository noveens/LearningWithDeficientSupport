import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from model import RegressionModelCifar
from utils import *

class DataLoader():
    def __init__(
        self, hyper_params, transforms, data, x, 
        delta, prop = None, action = None, rewards = None, 
        full_info = False, train_regressor = False,
        test_set = False, test_unsupp_actions = None
    ):
        self.x = x
        self.data = data
        self.delta = delta
        self.prop = prop
        self.action = action
        self.rewards = rewards
        self.bsz = hyper_params['batch_size']
        self.hyper_params = hyper_params
        self.transforms = transforms
        self.test_unsupp_actions = test_unsupp_actions
        if not (full_info or train_regressor or test_set): 
            self.precompute_rewards()

    def __len__(self):
        return len(self.x)

    def precompute_rewards(self):
        if self.rewards is None:
            self.rewards = []

            file_path  = '../regression_models/' + str(self.hyper_params['dataset']) + '/'
            file_path += 'pretrained_on_' + str(self.hyper_params['pretrained_on']) + '_'
            file_path += 'num_sample_' + str(self.hyper_params['num_sample']) + '_'
            file_path += 'tao_' + str(self.hyper_params['temperature']) + '_'
            file_path += 'clip_threshold_' + str(self.hyper_params['clip_threshold'])  + '_'
            file_path += 'features_' + str(int(self.hyper_params['dm_features']))
            file_path += '.pt'

            model = RegressionModelCifar({ 'dm_features' : self.hyper_params['dm_features'] }).cuda()
            model.load_state_dict(torch.load(file_path))
            model.eval()

            with torch.no_grad():
                print("Pre-computing rewards...")
                for ind in tqdm(range(len(self.data))):
                    image = self.normalize(self.data[ind])
                    image = Variable(FloatTensor([ image ]))
                    temp = model(image).data[0].cpu().numpy().tolist()
                    self.rewards.append(temp)

    def normalize(self, data_batch):
        transformed = self.transforms(
            np.array(data_batch * 255.0).astype('uint8').reshape(3, 32, 32).transpose(1, 2, 0)
        )
        return transformed.numpy().tolist()
        
    def iter(self, test_set = False):
        x_batch, y_batch = [], []
        action, delta, prop, regressed_rewards = [], [], [], []
        all_delta, all_prop, unsupp_actions = [], [], []

        self.loop = tqdm(range(len(self.x)))
        for ind in self.loop:
            data_point = self.data[int(self.x[ind])]

            if test_set == True: 
                x_batch.append(self.normalize(data_point))
                y_batch.append(np.argmax(self.delta[ind]))
                all_delta.append(self.delta[ind])
                if self.test_unsupp_actions is not None:
                    unsupp_actions.append(self.test_unsupp_actions[ind])

                if len(x_batch) == self.bsz:
                    yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch)), all_delta, unsupp_actions
                    x_batch, y_batch, all_delta, unsupp_actions = [], [], [], []

                continue

            # Floating point errors
            if self.prop[ind][self.action[ind]] < 0.0001: continue

            x_batch.append(self.normalize(data_point))
            y_batch.append(np.argmax(self.delta[ind]))
            action.append(self.action[ind])
            
            prop.append(self.prop[ind][self.action[ind]])
            all_prop.append(self.prop[ind])

            if self.delta[ind][self.action[ind]] == -12345: delta.append(0.0)
            else: delta.append(self.delta[ind][self.action[ind]])
            all_delta.append(self.delta[ind])
            if self.rewards is not None: 
                regressed_rewards.append(self.rewards[int(self.x[ind])])

            if len(x_batch) == self.bsz:
                yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch)), Variable(LongTensor(action)), \
                Variable(FloatTensor(delta)), Variable(FloatTensor(prop)), Variable(FloatTensor(all_prop)), all_delta, \
                Variable(FloatTensor(regressed_rewards))
                
                x_batch, y_batch = [], []
                action, delta, prop, regressed_rewards = [], [], [], []
                all_delta, all_prop = [], []

    def iter_full_info(self, eval = False):
        x_batch, y_batch, all_delta = [], [], []
        
        self.loop = tqdm(range(len(self.x)))
        for ind in self.loop:
            if eval == False and ind >= self.hyper_params['train_limit']: 
                self.loop.close()
                break

            data_point = self.data[int(self.x[ind])]

            x_batch.append(self.normalize(data_point))
            y_batch.append(np.argmax(self.delta[ind]))
            all_delta.append(self.delta[ind])

            if len(x_batch) == self.bsz:
                if eval == True:
                    yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch)), all_delta
                else:
                    yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch))
                x_batch, y_batch, all_delta = [], [], []

def readfile(path, hyper_params):
    x, delta, prop, action = [], [], [], []
    
    data = load_obj(path)
    
    for line in data:    
        x.append(line[0])
        delta.append(line[1:11])
        prop.append(line[11:21])
        action.append(int(line[-1]))

    return np.array(x), np.array(delta), np.array(prop), np.array(action)

def readfile_full_info(path):
    data = load_obj(path)
    x, delta = [], []    
    for i in data:
        x.append(np.array(i[0]).astype(float))
        delta.append(np.array(i[1]).astype(float))
    return np.array(x), np.array(delta)

def load_data(hyper_params, train_regressor = False):

    # Load x data
    path  = '../data/' + str(hyper_params['dataset']) + '/x'
    data_train = load_obj(path + '_train')
    data_test = load_obj(path + '_test')
    data_val = load_obj(path + '_train')

    # Load bandit data
    path  = '../data/' + str(hyper_params['dataset']) + '/bandit_data/' 
    path += 'pretrained_on_' + str(hyper_params['pretrained_on']) + '_'
    path += 'tao_' + str(hyper_params['temperature']) + '_'
    path += 'sampled_' + str(hyper_params['num_sample']) + '_'
    path += 'clip_threshold_' + str(hyper_params['clip_threshold'])

    x_train, delta_train, prop_train, action_train = readfile(path + '_train', hyper_params)
    x_val, delta_val, prop_val, action_val = readfile(path + '_val', hyper_params)

    path_full_info = '../data/' + str(hyper_params['dataset']) + '/' + 'full_information_data_test'
    x_test, delta_test = readfile_full_info(path_full_info)
    test_unsupp_actions = None
    # Load unsupported action list if we need to prune them while testing
    if hyper_params['prune_unsupported'] in [ 'testing', 'both' ]:
        print("Loading unsupported actions under the logging policy ON THE TEST SET..")
        test_unsupp_actions = load_obj(path + "_test_unsupp_actions")

    if hyper_params['imputed'] != 'none':
        print("Loading " + hyper_params['imputed'] + " imputed bandit data...")
        
        impute_file_path  = '../data/' + str(hyper_params['dataset']) + '/imputed_data/' 
        impute_file_path += 'pretrained_on_' + str(hyper_params['pretrained_on']) + '_'
        impute_file_path += 'tao_' + str(hyper_params['temperature']) + '_'
        impute_file_path += 'sampled_' + str(hyper_params['num_sample']) + '_'
        impute_file_path += 'clip_threshold_' + str(hyper_params['clip_threshold'])
        
        x_imputed, delta_imputed, prop_imputed, action_imputed = readfile(impute_file_path, hyper_params)
        
        if len(x_imputed) > 0:
            x_train = np.concatenate((x_train, x_imputed), axis=0)
            delta_train = np.concatenate((delta_train, delta_imputed), axis=0)
            prop_train = np.concatenate((prop_train, prop_imputed), axis=0)
            action_train = np.concatenate((action_train, action_imputed), axis=0)
    
    # Shuffle training data 
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    limit = hyper_params['train_limit']
    if hyper_params['imputed'] != 'none': limit *= 2
    indices = indices[:limit]
    
    x_train = x_train[indices]
    delta_train = delta_train[indices]
    prop_train = prop_train[indices]
    action_train = action_train[indices]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49139968, 0.48215827, 0.44653124), 
            (0.24703233, 0.24348505, 0.26158768)
        )
    ])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49139968, 0.48215827, 0.44653124), 
            (0.24703233, 0.24348505, 0.26158768)
        )
    ])

    trainloader = DataLoader(
        hyper_params, train_transforms, data_train, 
        x_train, delta_train, prop_train, action_train, 
        train_regressor = train_regressor
    )

    # Since train and val data have same images but different indices
    # We can re-use the computed rewards
    valloader = DataLoader(
        hyper_params, test_transforms, data_val, 
        x_val, delta_val, prop_val, action_val, 
        rewards = trainloader.rewards, train_regressor = train_regressor
    )
    testloader = DataLoader(
        hyper_params, test_transforms, data_test, 
        x_test, delta_test, test_set = True, 
        test_unsupp_actions = test_unsupp_actions
    )

    return trainloader, testloader, valloader

def load_data_full_info(hyper_params):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49139968, 0.48215827, 0.44653124), 
            (0.24703233, 0.24348505, 0.26158768)
        )
    ])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49139968, 0.48215827, 0.44653124), 
            (0.24703233, 0.24348505, 0.26158768)
        )
    ])

    path  = '../data/' + str(hyper_params['dataset']) + '/x'
    data_train = load_obj(path + '_train')
    data_test = load_obj(path + '_test')

    path = '../data/' + str(hyper_params['dataset']) + '/' + 'full_information_data_train'
    x, delta = readfile_full_info(path)
    trainloader = DataLoader(hyper_params, train_transforms, data_train, x, delta, full_info = True)

    path = '../data/' + str(hyper_params['dataset']) + '/' + 'full_information_data_test'
    x, delta = readfile_full_info(path)
    testloader = DataLoader(hyper_params, test_transforms, data_test, x, delta, full_info = True)

    return trainloader, testloader
