################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    q_2_4 = 10 ** np.linspace(-6, 2, 9) # question 2.4
    q_2_5 = [[128],[256,128],[512, 256, 128]] # question 2.5

    results = {'q_2_4': {'train accuracy': [],
                         'validation accuracy': [],
                         'test accuracy': [],
                         'loss': [],
                         'conf_mat': []},

               'q_2_5': {'train accuracy': [],
                         'validation accuracy': [],
                         'test accuracy': [],
                         'loss': [],
                         'conf_mat': []}}

    for learning_rate in q_2_4:
        epochs = 10
        lr = learning_rate
        hidden_dims = [128]
        batch_size = 128
        use_batch_norm = False
        dir = "data/"
        seed = 42
        print(f'learning_rate {lr}')
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, dir)
        results['q_2_4']['train accuracy'].append(logging_info['train accuracies'])
        results['q_2_4']['validation accuracy'].append(val_accuracies)
        results['q_2_4']['test accuracy'].append(test_accuracy)
        results['q_2_4']['loss'].append(logging_info['loss'])
        results['q_2_4']['confusion matrix'].append(logging_info['conf_mat'])
    for hidden_dims in q_2_5:
        print(f'hidden dims {hidden_dims}')
        lr = 0.1
        epochs = 20
        batch_size = 128
        use_batch_norm = True
        dir = "data/"
        seed = 42
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, dir)
        results['q_2_5']['train accuracy'].append(logging_info['train accuracies'])
        results['q_2_5']['validation accuracy'].append(val_accuracies)
        results['q_2_5']['test accuracy'].append(test_accuracy)
        results['q_2_5']['loss'].append(logging_info['loss'])
        results['q_2_5']['confusion matrix'].append(logging_info['conf_mat'])


    with open(results_filename, "a") as f:
        f.write(results)
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pass
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.txt' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)
