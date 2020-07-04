import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import datetime as dt
import time
from tensorboardX import SummaryWriter
writer = None

from model import ModelCifar, RegressionModelCifar
from data import load_data, load_data_full_info
from eval import evaluate
from get_k_value_model_selection import get_all
from loss import CustomLoss, MSELoss
from utils import *

def train(model, criterion, optimizer, reader, hyper_params, epoch):
    # Log the current loss to tensorboard 10 times per epoch
    log_after = int(len(reader) // (10.0 * hyper_params['batch_size']))

    metrics = {}
    metrics['Accuracy'] = 0.0
    total_batches = 0.0
    total_loss = 0.0

    model.train()

    for x, y, action, delta, prop, all_prop, all_delta, regressed_rewards in reader.iter():
        
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        output = model(x)
        if hyper_params['prune_unsupported'] in ['training', 'both']: output[all_prop == 0.0] = -1e7
        output = F.softmax(output, dim = 1)
        
        # Backward pass
        loss = criterion(output, y, action, delta, prop, all_prop, regressed_rewards)
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        # Fast Accuracy calculation trick
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.eq(y.data).unsqueeze(-1).float()
        correct = torch.matmul(predicted.t(), predicted)[0][0]
        metrics['Accuracy'] += 100.0 * correct / y.size(0)
        
        # Logging mechanism
        if int(total_batches) % log_after == 0 or int(total_batches + 1) == len(reader):
            log = float(total_loss / max(total_batches, 1.0))
            reader.loop.set_description(
                "epoch: {:3d} | loss: {:.4f} | Train Acc: {:.2f}".format(
                    epoch, log, metrics['Accuracy'] / max(total_batches, 1.0)
                )
            )
            num_steps = ((epoch - 1) * 10) + int(float(total_batches) // float(log_after))
            writer.add_scalar('loss', log, num_steps)
        
        total_batches += 1.0

    metrics['Loss'] = float(total_loss)
    for m in metrics: metrics[m] = round(float(metrics[m]) / total_batches, 4)

    return metrics

def train_full_info(model, criterion, optimizer, reader, hyper_params, epoch):
    log_after = len(reader) // (10.0 * hyper_params['batch_size'])

    total_loss = 0
    correct, total = 0.0, 0.0
    batch = 0

    model.train()

    for x, y in reader.iter_full_info():       
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        output = model(x)
        
        # Backward pass
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        # Computing Accuracy
        _, predicted = torch.max(output.data, 1)
        total += float(y.size(0))
        correct += float(predicted.eq(y.data).cpu().sum())

        # Logging
        if int(batch) % log_after == 0:
            log = float(total_loss / max(batch, 1))
            reader.loop.set_description(
                "epoch: {:3d} | loss: {:.4f} | Train Acc: {:.2f}".format(
                    epoch, log, correct * 100.0 / total
                )
            )

        batch += 1

    metrics = {}
    metrics['Accuracy'] = round(correct * 100.0 / total, 4)
    return metrics

def main(hyper_params = None, pretrain_full_info = False, train_regressor = False):
    # If custom hyper_params are not passed, load from hyper_params.py
    if hyper_params is None: from hyper_params import hyper_params
    else: print("Using passed hyper-parameters..")

    # Initialize a tensorboard writer
    global writer
    path = hyper_params['tensorboard_path']
    writer = SummaryWriter(path, flush_secs=20)

    # Loading data
    if pretrain_full_info == True: 
        train_reader, test_reader = load_data_full_info(hyper_params)
    else: 
        train_reader, test_reader, val_reader = load_data(hyper_params, train_regressor = train_regressor)
        hyper_params['all_ks'] = get_all(train_reader) # For MinSup evaluation

    file_write(hyper_params, "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params, "Data reading complete!")
    file_write(hyper_params, "Number of train batches: {:4d}".format(len(train_reader)))
    if pretrain_full_info == False: file_write(hyper_params, "Number of val batches: {:4d}".format(len(val_reader)))
    file_write(hyper_params, "Number of test batches: {:4d}".format(len(test_reader)))
    if 'all_ks' in hyper_params: file_write(hyper_params, "MinSup estimated k: " + str(hyper_params['all_ks']) + "\n\n")

    # Creating model
    if train_regressor: model = RegressionModelCifar(hyper_params)
    else: model = ModelCifar(hyper_params)
    if is_cuda_available: model.cuda()

    # Loss function
    if pretrain_full_info: criterion = nn.CrossEntropyLoss()
    elif train_regressor: criterion = MSELoss(hyper_params)
    else: criterion = CustomLoss(hyper_params)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyper_params['lr'], 
        momentum=0.9, weight_decay=hyper_params['weight_decay']
    )

    file_write(hyper_params, str(model))
    if pretrain_full_info == True: file_write(hyper_params, "Pre-training model on full information..")
    file_write(hyper_params, "\nModel Built!\nStarting Training...\n")

    best_metrics_val = None
    validate_on = hyper_params['validate_using'] # Estimator to chose best model (validation)
    if pretrain_full_info == True: validate_on = "Accuracy" # Since full-information

    try:
        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            metrics_train, metrics_val = None, None

            if pretrain_full_info == True:
                metrics_train = train_full_info(model, criterion, optimizer, train_reader, hyper_params, epoch)
                # Note that the metrics_train calculated is different from the actual model performance
                # Because the accuracy is calculated WHILE IT IS BEING TRAINED.
                # If we were to re-calculate the model performance keeping model parameters fixed:
                # we would get a different (most likely better) Accuracy.

                # Don't validate for logging policy. Just store the model at every epoch.
                torch.save(model.state_dict(), hyper_params['model_file'])
            else:
                metrics_train = train(model, criterion, optimizer, train_reader, hyper_params, epoch)
                # Calulating the metrics on the validation set
                metrics_val = evaluate(
                    model, criterion, val_reader, hyper_params, 
                    eval_estimators = True, test_set = False
                )
                
                # Validate
                if best_metrics_val is None: best_metrics_val = metrics_val
                elif metrics_val[validate_on] >= best_metrics_val[validate_on]: 
                    best_metrics_val = metrics_val
                
                # Save model if current is best epoch
                if metrics_val[validate_on] == best_metrics_val[validate_on]: 
                    torch.save(model.state_dict(), hyper_params['model_file'])

            metrics_train = None # Don't print train metrics, since already printing in tqdm bar
            log_end_epoch(hyper_params, epoch, epoch_start_time, writer, metrics_train, metrics_val)
            
    except KeyboardInterrupt: print('Exiting from training early')

    # Evaluate best saved model
    model = ModelCifar(hyper_params)
    if is_cuda_available: model.cuda()
    model.load_state_dict(torch.load(hyper_params['model_file']))
    model.eval()

    metrics_train = None
    metrics_test = evaluate(
        model, criterion, test_reader, hyper_params, 
        eval_estimators = False, test_set = True
    )

    file_write(hyper_params, "Final model performance on test-set:")
    log_end_epoch(
        hyper_params, hyper_params['epochs'] + 1, time.time(), 
        writer, metrics_train, metrics_test, test = True
    )
    
    writer.close()

    return metrics_test

if __name__ == '__main__':
    main()
