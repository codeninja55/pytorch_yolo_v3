#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
A PyTorch implementation of YOLOv3 Object Detection model.
Based heavily on implementation from
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/.
"""

# Built-in/Generic Imports
from typing import Tuple, List

# Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import ModuleList

__author__ = 'Andrew Che (codeninja55)'
__copyright__ = 'Copyright 2018, CN55 PyTorch YOLOv3'
__credits__ = ['Joseph Redmon, Ayoosh Kathuria']
__license__ = 'GNU Public License V3'
__version__ = '0.1.dev'
__maintainer__ = 'Andrew Che (codeninja55)'
__email__ = 'andrew@codeninja55.me'
__status__ = 'development'


# noinspection PyAbstractClass
class EmptyLayer(nn.Module):
    """
    Used to as a dummy layer in place of a proposed route and shortcut layer. We can perform the concatenation of these
    layers directly in the forward function of the nn.Module object representing darknet.
    """
    # Note: In PyTorch, when we define a new layer, we subclass nn.Module and write the operation the layer performs
    #       in the forward function.
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfgfile: str) -> List:
    """
    Takes a configuration file and parses it.
    :param cfgfile: The configuration file to pass.
    :return: A list of blocks. Each blocks describes a block in the neural network to be built. Block is represented
             as a dictionary in the list.
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 # store the lines in a list
    lines = [x for x in lines if len(x) > 0]        # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']       # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]    # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":                          # this marks the start of a new block
            if len(block) != 0:                     # if blk is not empty, implies it is storing values of previous blk
                blocks.append(block)                # add it in the blocks list
                block = {}                          # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks: List) -> Tuple:
    """
    Create all the nn.Module objects based on the configurations passed from the yolov3 cfg file in a sequential manner.
    From this, each block in the list we will create the multiple layers required.
    :param blocks: a list of blocks represented as a dict read from the cfg file.
    :return: A tuple with the net_info information about input and pre-processing and a nn.ModuleList()
    """
    net_info = blocks[0]            # captures the information about the input and pre-processing
    module_list: ModuleList = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    filters = 0

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        if x["type"] == "convolutional":
            # get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except Exception as e:
                print("[***DEBUG***] Exception thrown in create_modules() \n{}".format(e.__cause__))
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            pad = (kernel_size - 1) // 2 if padding else 0

            # add the Convolutional Layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check the activation
            # it is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        # if its an upsampling layer - we use Bilinear2dUpsampling
        elif x["type"] == "upsample":
            stride = int(x["stride"])

            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)
        # if it is a route layer
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            try:    # end if there exists one
                end = int(x["layers"][1])
            except Exception as e:
                print("[***DEBUG***] Exception thrown in create_modules() \n{}".format(e.__cause__))
                end = 0

            # positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # The convolutional layer just in front of a route layer applies it's kernel to (possibly concatenated)
            # feature maps from a previous layers. The following code updates the filters variable to hold the number
            # of filters outputted by a route layer.
            if end < 0:
                # if we are concatenating maps
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connections
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)
        # YOLO is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]    # should be a total of 9
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
        return net_info, module_list


# TODO: Test
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks=blocks))
