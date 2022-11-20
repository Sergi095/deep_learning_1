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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.size = (in_features, out_features)
        self.params = {}
        self.grads = {}
        self.in_linear_mod = 0
        self.in_elu_mod = 0
        self.out_ = 0
        # Kaiming ini as seen in tutorial 4 notebook
        if input_layer:
            # initialize with W~N(0,1/np.sqrt(in_features))
            self.params['weight'] = np.random.normal(loc=0.0,
                                                     scale=1/np.sqrt(in_features),
                                                     size=self.size)
        else:
            # initialize with W~N(0,np.sqrt(2)/np.sqrt(in_features))
            self.params['weight'] = np.random.normal(loc=0.0,
                                                     scale=np.sqrt(2)/np.sqrt(in_features),
                                                     size=self.size)

        self.grads['weight'] = np.zeros((in_features, out_features))
        self.params['bias'] = np.zeros((out_features,))
        self.grads['bias'] = np.zeros((in_features,))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_linear_mod = x.copy()
        # out = (x * W + b)
        out = self.in_linear_mod @ self.params['weight'] + self.params['bias']
        # print(f'forward out {out.shape}')
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['bias'] = np.mean(dout, axis=0)
        dx = dout @ self.params['weight'].T
        self.grads['weight'] = self.in_linear_mod.T @ dout
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_linear_mod = None
        self.grads['weight'] = None
        self.grads['bias'] = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.in_elu_mod = x.copy()
        out = np.where(x > 0,
                       x, # if x>0
                       np.exp(x)-1) # if x<=0
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = np.where(self.in_elu_mod > 0,
                      np.multiply(dout, 1), # if x>0
                      np.multiply(dout, np.exp(self.in_elu_mod))) #if x<=0
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_elu_mod = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        mu = x.max(axis=1, keepdims=True)
        X = x - mu
        out = np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)
        self.out_ = out.copy()
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        prod_ = np.einsum('ij,ik->ijk', self.out_, self.out_)
        grad_ = np.subtract(np.einsum('ij,jk->ijk', self.out_, np.eye(self.out_.shape[1])), prod_)
        dx = np.einsum('ijk, ik -> ij', grad_, dout)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx
    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.out_ = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        T = np.identity(x.shape[1])[y]
        out = -1/ x.shape[0] * np.sum(T * np.log(x))
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        T = np.identity(x.shape[1])[y]
        dx = -1/x.shape[0] * (T / x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
