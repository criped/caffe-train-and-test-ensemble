# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:03:07 2018

@author: cristian
"""

# Set up the Python environment: we'll use the pylab import for numpy and plot inline.
from pylab import *
from . import base_solver
import numpy as np
import sys

# Import caffe, adding it to sys.path if needed. Make sure you've built pycaffe.
caffe_root = '<caffe_root_path>'  # this file should be run from {caffe_root}/examples (otherwise change this line)

sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# Dictionary with the parameters to tune the base solver
param_values = dict()
# param_values['type'] = ['SGD', 'Adam']
# param_values['base_lr'] = [0.001, 0.0001]
# param_values['momentum'] = [0.9]
# param_values['weight_decay'] = [0.0004, 0.004]
# param_values['weight_decay'] = [0.0004]

net = '<path_to_caffe_models_folder>/BreaKHis_cnn4_train_test.prototxt'
n_nets_to_train = 1
niter = 50000
snapshot_it = 50000
test_forward_steps = 100
accuracy_layer = 'accuracy'
loss_layer = 'loss'

snapshot_prefix_path = '<path_to_snapshots_folder>/<snapshot_prefix>'
base_plot_path = '<path_to_plots_folder>'
base_solver_path = '<path_to_solvers_folder>'


def train_cnn(solver_config_path, s, output_plot_path):    
    
    # load the solver and create train and test nets
    solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.get_solver(solver_config_path)

    test_interval = s.test_interval
    
    # losses will also be stored in the log
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))
    test_loss = zeros(int(np.ceil(niter / test_interval)))
    
    print("niter {}, test_interval {}".format(niter, test_interval))

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        # store the train loss
        train_loss[it] = solver.net.blobs[loss_layer].data
       
        if it % test_interval == 0:
            print('Iteration', it, 'testing...')
            for test_it in range(test_forward_steps):
                solver.test_nets[0].forward()

            test_acc[it // test_interval] = solver.test_nets[0].blobs[accuracy_layer].data
            test_loss[it // test_interval] = solver.test_nets[0].blobs[loss_layer].data
            
            print("Accuracy calculated = {} ; accuracy layer = {}"
                  .format(test_acc[it // test_interval], solver.test_nets[0].blobs[accuracy_layer].data)
                  )
    
    fig, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'g')
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss (red), test loss (green)')
    ax2.set_ylabel('test accuracy (blue)')
    ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
    
    fig.savefig(output_plot_path)


def train_values_loop(param_val, solver_net, output_plot_path, it, solver_config_path):
    for key, values_list in param_val.iteritems():
        for value in values_list:
            print('Training updating {}: {}'.format(key, value))
            
            s = base_solver.default_SGDsolver(solver_net)
            setattr(s, key, value)
            s.snapshot_prefix = (snapshot_prefix_path)
            s.snapshot = snapshot_it
    
            if not output_plot_path:
                generated_plot_path = "{}{}_{}_{}iters.jpg".format(base_plot_path, key, value, it)
                train_cnn(solver_config_path, s, generated_plot_path)
                
            else:                        
                train_cnn(solver_config_path, s, output_plot_path)


if __name__ == '__main__':
    for i in range(n_nets_to_train):
        solver_path = base_solver_path + '/BreaKHis_cnn_solver_{}.prototxt'.format(i)
        print("Iteration: {}".format(i))

        train_values_loop(param_val=param_values, solver_net=net,
                          output_plot_path=None, it=i, solver_config_path=solver_path)

