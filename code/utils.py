import torch
import json
import pickle
import time

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_json(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)

def file_write(hyper_params, s):
    if 'dont_print' in hyper_params and hyper_params['dont_print'] == False: print(s)
    f = open(hyper_params['log_file'], 'a')
    f.write(s+'\n')
    f.close()

def log_end_epoch(hyper_params, epoch, epoch_start_time, writer, metrics_train, metrics_test, test = False):
    log_string = ""

    if metrics_train is not None:
        for metric in metrics_train: 
            writer.add_scalar('Train_metrics/' + metric, metrics_train[metric], epoch - 1)

        log_string += '-' * 89
        log_string += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        for m in metrics_train: log_string += " | " + m + ' = ' + str(metrics_train[m])
        log_string += ' (TRAIN)\n'
        log_string += '-' * 89

    if metrics_test is not None:
        for metric in metrics_test: 
            writer.add_scalar('Test_metrics/' + metric, metrics_test[metric], epoch - 1)

        log_string += '-' * 89
        log_string += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        for m in metrics_test: log_string += " | " + m + ' = ' + str(metrics_test[m])
        if test == True: log_string += ' (TEST)\n'
        else: log_string += ' (VAL)\n'
        log_string += '-' * 89
    
    file_write(hyper_params, log_string)

def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()

def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')
