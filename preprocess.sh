#!/bin/bash

GPU_ID=0 # Default GPU ID
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then GPU_ID=$CUDA_VISIBLE_DEVICES; fi

# Settings to pre-process for (comma-seperated for multiple settings)
pretrain_on="35000" # Logging policy trained on these many FULL-INFORMATION contexts
pretrain_epochs="2" # Logging policy trained for these many epochs. NO COMMA SUPPORT.
dm_features="32" # Train regression function on these many image features. Max: 32
num_sample="1" # Number of actions to sample per context
temperatures="4" # Controls support deficiency in logging policy. pi_o = pi_o ** temperature
clip_threshold="0.01" # Will set propensity = 0 for actions with prob < 0.01 (logging policy)

# Make model/log directories (if they don't exist)
mkdir -p data/cifar saved_logs saved_models regression_models/cifar
mkdir -p logging_policies/cifar data/cifar/bandit_data data/cifar/imputed_data

# Download CIFAR-10 dataset
cd data/cifar;
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;
tar -xvf cifar-10-python.tar.gz; rm cifar-10-python.tar.gz;
cd ../..

# Make full information data for pre-training logging policy
cd preprocessing_code; 
echo -e "\033[0;31mMaking full information data for pre-training logging policy..\033[0m"
CUDA_VISIBLE_DEVICES=$GPU_ID python preprocess_cifar.py $pretrain_on $temperatures $num_sample $clip_threshold "True"; 
cd ..

# Pre-train logging policy
cd preprocessing_code; 
echo -e "\033[0;31mPre-training logging policy..\033[0m"
CUDA_VISIBLE_DEVICES=$GPU_ID python pretrain_logging_policy.py $pretrain_on $pretrain_epochs; 
cd ..

# Make Bandit Feedback data
cd preprocessing_code; 
echo -e "\033[0;31mMaking bandit feedback data..\033[0m"
CUDA_VISIBLE_DEVICES=$GPU_ID python preprocess_cifar.py $pretrain_on $temperatures $num_sample $clip_threshold "False"; 
cd ..

# Train regression functions
cd preprocessing_code; 
echo -e "\033[0;31mTraining regression functions..\033[0m"
CUDA_VISIBLE_DEVICES=$GPU_ID python pretrain_regression_function.py $pretrain_on $temperatures $num_sample $dm_features $clip_threshold; 
cd ..

# Impute bandit feedback dataset for running 
# efficient approximation of the `conservative_extrapolation` method
cd preprocessing_code; 
echo -e "\033[0;31mNegative imputing bandit feedback data..\033[0m"
CUDA_VISIBLE_DEVICES=$GPU_ID python impute_cifar.py $pretrain_on $temperatures $num_sample $clip_threshold; 
cd ..
