################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_batches = predictions.shape[0]
    n_classes = predictions.shape[1]
    size_conf_mat = (n_classes, n_classes)

    conf_mat = torch.zeros(size_conf_mat)
    for batch in range(n_batches):
        preds_ = torch.argmax(predictions[batch])
        targs_ = targets[batch]
        conf_mat[preds_, targs_] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    TP_TN = torch.diag(confusion_matrix)
    metrics = {}
    metrics['accuracy'] = torch.sum(TP_TN) / torch.sum(confusion_matrix)
    metrics['precision'] = TP_TN / (torch.sum(confusion_matrix, 1))
    metrics['recall'] = TP_TN / (torch.sum(confusion_matrix, 0))
    metrics['f1_beta'] = (1 + beta**2) * (metrics['precision'] * metrics['recall']) \
                         / (beta**2 * metrics['precision'] + metrics['recall'])
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf_mat = torch.zeros((num_classes, num_classes))
    for input_data, targets in data_loader:
        batch_size = input_data.shape[0] # batch size
        n_features = np.prod(input_data.shape[1:])
                                        #128,       #32*32*3
        input_data = input_data.reshape((batch_size, n_features)) # X R^batch_Size X N
        # todevice
        input_data = input_data.to(device)
        predictions = model.forward(input_data)
        conf_mat += confusion_matrix(predictions, targets)
    metrics = confusion_matrix_to_metrics(conf_mat)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics, conf_mat


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = len(cifar10_loader['train'].dataset.dataset.classes)
    n_features = 1
    for elem in cifar10_loader['train'].dataset.dataset.data.shape[1:]:
        n_features *= elem
    # TODO: Initialize model and loss module
    model = MLP(n_inputs=n_features,
                n_hidden=hidden_dims,
                n_classes=n_classes,
                use_batch_norm=use_batch_norm)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr)
    # TODO: Add any information you might want to save for plotting
    logging_info = {}
    # TODO: Training loop including validation
    models = {}
    val_accuracies_epoch = {}
    train_accuracies_epoch  = {}
    epoch_losses = {}
    # Simplified training loop
    for epoch in range(epochs):
        epoch = epoch + 1
        print(f'epoch #{epoch}')
        epoch_loss = [] # epoch loss
        for input_data, labels in tqdm(cifar10_loader['train']):  # looping for each batch

            # reshaping inputs
            batch_size = input_data.shape[0] # batch size
                                            #128,       #32*32*3
            input_data = input_data.reshape((batch_size, n_features)) # X R^batch_Size X N
            # todevice
            input_data = input_data.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            # forward
            out = model.forward(input_data)
            loss = loss_module.forward(out, labels)
            epoch_loss.append(loss.cpu().detach().item())

            # backward
            loss.backward()
            # TODO: Do optimization with the simple SGD optimizer
            # mini-batch gradient descent
            opt.step()


        model_ = deepcopy(model)
        models[epoch] = model_
        # epoch evaluation
        train_epoch_metrics, _ = evaluate_model(model,
                                          cifar10_loader['train'],
                                          n_classes)

        val_epoch_metrics, _ = evaluate_model(model,
                                          cifar10_loader['validation'],
                                          n_classes)
        print('\n ')
        print(f"epoch #{epoch}, Train accuracy: {train_epoch_metrics['accuracy']}")
        print(f"epoch #{epoch}, Validation accuracy: {val_epoch_metrics['accuracy']}")
        print(f"epoch #{epoch}, mean loss: {np.mean(epoch_loss)}")
        print('\n ')

        train_accuracies_epoch[epoch] = train_epoch_metrics['accuracy']
        val_accuracies_epoch[epoch] = val_epoch_metrics['accuracy']
        epoch_losses[epoch] = np.mean(epoch_loss)

    logging_info['train accuracies'] = list(train_accuracies_epoch.values())
    logging_info['loss'] = list(epoch_losses.values())
    val_accuracies = list(val_accuracies_epoch.values())
    # TODO: Test best model
    best_model = max(val_accuracies_epoch, key=val_accuracies_epoch.get)
    print(f'Best model: epoch_{best_model}, with accuracy: {max(val_accuracies_epoch.values())}')
    print("TESTING ...")
    model = models[best_model]

    test_metrics, conf_mat = evaluate_model(model,
                                  cifar10_loader['test'],
                                  n_classes)
    logging_info['conf_mat'] = conf_mat
    for metrics, result in test_metrics.items():
        if metrics != 'accuracy':
            logging_info[metrics] = result
    test_accuracy = test_metrics['accuracy']
    print(f'Test Accuracy: {test_accuracy}')
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')


    args = parser.parse_args()
    kwargs = vars(args)
    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)



    # Feel free to add any additional functions, such as plotting of the loss curve here
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Loss curve')
    plt.plot(np.arange(0, args.epochs, 1), logging_info['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('pytorch mlp loss')
    plt.show()

    plt.figure()
    plt.title('Validation Accuracy curve')
    plt.plot(val_accuracies, label='validation_accuracies')
    plt.plot(logging_info['train accuracies'], label='Train_accuracies')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('pytorch mlp accuracy')
    plt.legend()
    plt.show()

    sns.heatmap(logging_info['conf_mat'], annot=True)
    plt.savefig('confusion matrix pytorch')
    plt.show()
