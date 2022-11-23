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
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set




def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=True)
    # Replace the last layer with a linear layer with num_classes outputs
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir=data_dir, augmentation_name=augmentation_name)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch
    best_accuracy = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('Epoch: %d - Loss: %f' % (epoch + 1, running_loss / len(train_loader)))

        accuracy = evaluate_model(model, val_loader, device)

        if accuracy > best_accuracy:  # Save the best model on validation accuracy and return it at the end of training process
            torch.save(model.state_dict(), checkpoint_name)
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    # Loop over the dataset and compute the accuracy. Return the accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        try:
            for data in data_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        except AttributeError:
            for data in data_loader:
                images, labels = data[0].unsqueeze(0).to(device), torch.tensor(data[1]).unsqueeze(0).to(device)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # Set model back to training mode
    model.train()
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)
    # Set the device to use for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = get_model().to(device)
    # Get the augmentation to use
    augmentation = augmentation_name
    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, 'best_model.pt', device, augmentation)
    # Evaluate the model on the test set
    test_set = get_test_set(data_dir)
    evaluate_model(model, test_set, device)
    accuracy = evaluate_model(model, test_set, device)
    print('Accuracy on the test set: %f' % accuracy)
    print(f'Finished Training for {augmentation}')
    print('\n \n \n')
    return accuracy
    #######################
    # END OF YOUR CODE    #
    #######################

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # # Feel free to add more arguments or change the setup
    #
    # parser.add_argument('--lr', default=0.001, type=float,
    #                     help='Learning rate to use')
    # parser.add_argument('--batch_size', default=128, type=int,
    #                     help='Minibatch size')
    # parser.add_argument('--epochs', default=30, type=int,
    #                     help='Max number of epochs')
    # parser.add_argument('--seed', default=123, type=int,
    #                     help='Seed to use for reproducing results')
    # parser.add_argument('--data_dir', default='data/', type=str,
    #                     help='Data directory where to store/find the CIFAR100 dataset.')
    # parser.add_argument('--augmentation_name', default=None, type=str,
    #                     help='Augmentation to use.')

    # args = parser.parse_args()
    # kwargs = vars(args)

    lr, batch_size, epochs, data_dir, seed = 0.001, 128, 30, 'data/', 123
    transform_name = ['random_horizontal_flip', 'random_crop', 'color_jitter', None]
    accuracy_dict = {}
    for augmentation_name in transform_name:
        accuracy = main(lr, batch_size, epochs, data_dir, seed, augmentation_name)
        accuracy_dict[str(augmentation_name)] = accuracy
    import pickle
    import matplotlib.pyplot as plt

    with open('accuracy_dict.pkl', 'wb') as f:
        pickle.dump(accuracy_dict, f)
    f.close()

    # plot the accuracy of different augmentation
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(accuracy_dict.keys(), accuracy_dict.values())
    plt.savefig('accuracy_augmentation.png')
    plt.show()





