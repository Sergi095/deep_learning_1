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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        #one layer, 128 hidden units, 10 epochs, learning rate 0.1, seed 42
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs # int with number of inputs
        self.n_hidden = n_hidden # [N1(#of units at layer_1), ... , Nn (#of units at layer_n)]
        self.n_classes = n_classes # int with number of classes
        # Empty list for layers
        self.layers = [] # [linear, elu, ..., linear, softmax]
        self.ELU = ELUModule()
        self.softmax = SoftMaxModule()

        hidden_dim = len(self.n_hidden)
        for i in range(hidden_dim+1):
            #first layer
            if i == 0:
                self.layers.append(LinearModule(self.n_inputs,
                                                   self.n_hidden[i],
                                                   True)) # Kaiming ini
                self.layers.append(self.ELU)
            #last layer
            elif i==hidden_dim:
                self.layers.append(LinearModule(self.n_hidden[-1],
                                                   self.n_classes,
                                                   False))
                self.layers.append(self.softmax)
            else:
                self.layers.append(LinearModule(self.n_hidden[i-1],
                                                   self.n_hidden[i],
                                                   False))
                self.layers.append(self.ELU)



        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers:
            layer.clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################
