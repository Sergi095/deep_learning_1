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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        # self.pad_up = nn.Parameter(torch.zeros(size=(1, 3, pad_size, image_size)))
        # self.pad_left = nn.Parameter(torch.zeros(size=(1, 3, pad_size + image_size, pad_size)))
        # self.pad_down = nn.Parameter(torch.zeros(size=(1, 3, pad_size, image_size)))
        # self.pad_right = nn.Parameter(torch.zeros(size=(1, 3, pad_size + image_size, pad_size)))

        self.pad_up = nn.Parameter(torch.randn(size=(1, 3, pad_size, image_size), requires_grad=True))
        self.pad_down = nn.Parameter(torch.randn(size=(1, 3, pad_size, image_size),requires_grad=True))
        self.pad_right = nn.Parameter(torch.randn(size=(1, 3, -2 * pad_size + image_size, pad_size),requires_grad=True))
        self.pad_left = nn.Parameter(torch.randn(size=(1, 3, -2 * pad_size + image_size, pad_size),requires_grad=True))
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        # x = torch.cat((self.pad_left, x, self.pad_right), dim=3)
        # x = torch.cat((self.pad_up, x, self.pad_down), dim=2)

        print(x.shape)
        x_ = torch.zeros_like(x)
        pad_size = self.pad_up.shape[2]
        image_size = self.pad_up.shape[3]
        x_[:, :, 0:pad_size, :] = self.pad_up
        x_[:, :, -pad_size:x_.shape[-2], :] = self.pad_down
        x_[:, :, 0:-2 * pad_size + image_size, 0:pad_size] = self.pad_right
        x_[:, :, 0:-2 * pad_size + image_size, -pad_size:x_.shape[-1]] = self.pad_left
        x += x_
        return x

        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.patch = nn.Parameter(torch.randn(size=(1, 3, args.prompt_size, args.prompt_size)))
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        x[:, :, :self.patch.shape[2], :self.patch.shape[3]] += self.patch
        return x
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn
        self.patch = nn.Parameter(torch.randn(size=(1, 3, args.prompt_size, args.prompt_size)))
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        rand_x = np.random.randint(0, x.shape[2] - self.patch.shape[2])
        rand_y = np.random.randint(0, x.shape[3] - self.patch.shape[3])
        x[:, :, rand_x:rand_x+self.patch.shape[2], rand_y:rand_y+self.patch.shape[3]] += self.patch
        return x
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

