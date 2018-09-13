# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:22:44 2018

@author: cristian

It allows to setup a solver configuration programatically.
Receives the prototext file path incluing the network structure.
Outputs the configured solver.
"""

from caffe.proto import caffe_pb2


def default_SGDsolver(net_path):
    s = caffe_pb2.SolverParameter()
    
    # Set a seed for reproducible experiments:
    # this controls for randomization in training.
#    s.random_seed = 0xCAFFE
    
    # Specify locations of the train and (maybe) test networks.
#    s.train_net = train_net_path
#    s.test_net.append(test_net_path)
    s.net = net_path
    
    s.test_interval = 500  # Test after every 500 training iterations.
    s.test_iter.append(100) # Test on 100 batches each time we test.
    
#    s.max_iter = 50000     # no. of times to update the net (training iterations)
    s.max_iter = 10000    # no. of times to update the net (training iterations)
     
    # EDIT HERE to try different solvers
    # solver types include "SGD", "Adam", and "Nesterov" among others.
    s.type = "SGD"
    
    # Set the initial learning rate for SGD.
    s.base_lr = 0.0001  # EDIT HERE to try different learning rates
    # Set momentum to accelerate learning by
    # taking weighted average of current and previous updates.
    s.momentum = 0.9
    # Set weight decay to regularize and prevent overfitting
    s.weight_decay = 0.004
    
    # Set `lr_policy` to define how the learning rate changes during training.
    # This is the same policy as our default LeNet.
    s.lr_policy = 'fixed'
#    s.gamma = 0.0001
#    s.power = 0.75
    # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
    # `fixed` is the simplest policy that keeps the learning rate constant.
    # s.lr_policy = 'fixed'
    
    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000
    
    # Snapshots are files used to store networks we've trained.
    # We'll snapshot every 50K iterations -- once during training.
    s.snapshot = 10000
    s.snapshot_prefix = '<snapshot_storage_path>/<snapshot_prefix>'
    
    # Train on the GPU
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    return s
