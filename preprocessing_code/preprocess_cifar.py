import numpy as np
import pickle
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import sys
sys.path.insert(0, '../code')

from model import ModelCifar
from utils import FloatTensor, save_obj

pretrained_on_list = list(map(int, sys.argv[1].split(",")))
taos_list = list(map(int, sys.argv[2].split(",")))
num_sample_list = list(map(int, sys.argv[3].split(",")))
clip_threshold_list = list(map(float, sys.argv[4].split(",")))
full_information_only = sys.argv[5] == "True"

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

def one_hot(arr):
    new = []
    for i in range(len(arr)):
        temp = np.zeros(10)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def clip_and_renorm(probs, clip_threshold):
    # Clip the ones less than threshold to zero
    mask = probs <= clip_threshold
    num_z = float(torch.sum(mask))
    probs[mask] = 0.0

    # L1 Re-normalize
    probs = probs / torch.sum(probs)

    # Re-normalize fix
    probs = probs.cpu().numpy()
    probs[np.argmax(probs)] += 0.9995 - float(np.sum(probs))
    return probs, num_z

def normalize(data_batch):
    transformed = train_transforms(
        np.array(data_batch * 255.0).astype('uint8').reshape(3, 32, 32).transpose(1, 2, 0)
    )
    return transformed.numpy().tolist()

def get_probs(logging_policy, x, clip_threshold, tao):
    # Forward pass on logging policy
    logging_output = logging_policy(Variable(FloatTensor([ normalize(x) ]))).data[0]

    # Do a temperature softmax
    logging_output = logging_output * tao
    pi_o = F.softmax(logging_output, dim = 0)
    pi_o, num_z = clip_and_renorm(pi_o, clip_threshold)

    return pi_o, num_z

x_train, x_test = [], []
y_train, y_test = [], []

# Train
for b in range(1, 6):
    this_batch = unpickle("../data/cifar/cifar-10-batches-py/data_batch_" + str(b))

    if len(x_train) == 0: x_train, y_train = this_batch[b'data'], this_batch[b'labels']
    else: 
        x_train = np.concatenate((x_train, this_batch[b'data']), axis=0)
        y_train = np.concatenate((y_train, this_batch[b'labels']), axis=0)

# Test
this_batch = unpickle("../data/cifar/cifar-10-batches-py/test_batch")
x_test, y_test = this_batch[b'data'], this_batch[b'labels']

# Normalize X data
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0

# One hot the rewards
y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Save full information dataset
if full_information_only == True:
    full_data = list(map(lambda i: [ i, y_train[i].tolist() ] , range(len(x_train))))
    save_obj(full_data, '../data/cifar/full_information_data_train')

    full_data = list(map(lambda i: [ i, y_test[i].tolist() ] , range(len(x_test))))
    save_obj(full_data, '../data/cifar/full_information_data_test')

    filename  = '../data/cifar/x'

    save_obj(x_train, filename + '_train')
    save_obj(x_test, filename + '_test')
    # No need for validation data, since validation has same indices as train

    # Don't run below code of making bandit_feedback data
    exit(0)

for pretrained_on in pretrained_on_list:

    # Load the logging policy
    logging_policy = ModelCifar({}).cuda()
    logging_policy.load_state_dict(
        torch.load('../logging_policies/cifar/' + str(pretrained_on) + '.pt')
    )
    logging_policy.eval()

    for tao in taos_list:
        for clip_threshold in clip_threshold_list:
            print(
                "For logging policy pretrained on = " + str(pretrained_on) + \
                "; temperature = " + str(tao) + \
                "; clip threshold = " + str(clip_threshold)
            )

            # Pre-compute logging outputs
            precomputed_logging_outputs = []
            avg_num_zeros = 0.0
            correct, total = 0.0, 0.0

            for point_num in tqdm(range(x_train.shape[0])):
                image = x_train[point_num]
                label = np.argmax(y_train[point_num])

                # Forward pass on saved logging policy
                probs, num_z = get_probs(logging_policy, image, clip_threshold, tao)
                precomputed_logging_outputs.append(probs)
                avg_num_zeros += num_z

                # Computing Accuracy
                actionvec = np.random.multinomial(1, probs)
                action = np.argmax(actionvec)
                correct += float(int(action == label))
                total += 1.0

            avg_num_zeros /= float(x_train.shape[0])
            avg_num_zeros = round(avg_num_zeros, 4)
            acc = round(100.0 * correct / total, 2)

            print(
                "Average unsupported actions = " + str(avg_num_zeros * 10.0) + "%" \
                ", Train Accuracy =", acc, '%\n'
            )

            for num_sample in num_sample_list:

                final_x, final_y, final_actions, final_prop = [], [], [], []

                for point_num in range(x_train.shape[0]):
                    for _ in range(num_sample):

                        probs = precomputed_logging_outputs[point_num]
                        actionvec = np.random.multinomial(1, probs)
                        action = np.argmax(actionvec)
                        
                        final_x.append([ point_num ])
                        final_y.append(y_train[point_num])
                        final_actions.append([ action ])
                        final_prop.append(probs)

                # Save as CSV
                train = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)

                # Shuffle train set
                indices = np.arange(len(train))
                np.random.shuffle(indices)
                train = train[indices]

                # Split into validation set. Validation set is 10% of train set.
                split_point = int(float(len(train)) * 0.1)
                val = train[:split_point]
                train = train[split_point:]

                # Check actions which don't have support in test-set.
                # Will only be used when the prune unsupported action method is used.
                print("Computing unsupported actions on the test-set..")
                test_unsupp_actions = []
                for image in tqdm(x_test):
                    probs, _ = get_probs(logging_policy, image, clip_threshold, tao)
                    test_unsupp_actions.append(probs == 0.0)
                test_unsupp_actions = np.array(test_unsupp_actions)

                # Save bandit data
                filename  = '../data/cifar/bandit_data/' 
                filename += 'pretrained_on_' + str(pretrained_on) + '_'
                filename += 'tao_' + str(tao) + '_'
                filename += 'sampled_' + str(num_sample) + '_'
                filename += 'clip_threshold_' + str(clip_threshold)

                save_obj(train, filename + '_train')
                save_obj(val, filename + '_val')
                save_obj(test_unsupp_actions, filename + '_test_unsupp_actions')
