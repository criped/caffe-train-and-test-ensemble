# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:21:38 2018

@author: root
"""

#Set up the Python environment: we'll use the pylab import for numpy and plot inline.
from pylab import *

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2

import base_solver
import numpy as np
import time
import lmdb

#Import caffe, adding it to sys.path if needed. Make sure you've built pycaffe.
caffe_root = '<caffe_root_path>'  # this file should be run from {caffe_root}/examples (otherwise change this line)
tfm_root = '<working_directory_root>'

# Path to output file storing a plot with the training results. 
output_plot_path= tfm_root+'<output_plot_path>'

# Path to weights (generated by a snapshot) of each network composing the ensemble
weights = "<path_to_weights_directory>/BreaKHis_cnn_-{}_iter_50000.caffemodel"

#solver_path = '<path_to_solver>'

ip_softmax = 'ip3' # Name of fc-layer next to the final loss layer.
n_nets_in_ensemble = 10 # Number of nets in ensemble

# Network architecture
model_def_net = tfm_root + "brasileños/BreaKHis_cnn4_train_test.prototxt"

caffe.set_device(0)
caffe.set_mode_gpu()


"""
Receives a batch of images and sets it to the solver as the current batch,
where the net make predictions on.
It also receives the ground truth of the given batch.
"""    
def test_in_batch(net, img_batch, ground_truth_batch):
    if ( len(img_batch) != len(ground_truth_batch) ):
        raise ValueError('Las dimensiones de los batches no coinciden')
    
    arch_info = [(k, v.data.shape) for k, v in net.blobs.items()]
    print "Blobs dimensions {}".format(arch_info)
    time.sleep(10)
    for idx in range(len(img_batch)):
        net.blobs['data'].data[idx] = img_batch[idx]
        net.blobs['label'].data[idx] = ground_truth_batch[idx]
    
    output = net.forward()
    print "output", output
    time.sleep(5)
    return net.blobs[ip_softmax].data.argmax(1) 

	
def test_by_solver(solver, test_interval):
	forward_steps = 10
    for i in range(test_interval*forward_steps):
        solver.test_nets[0].forward()

    return solver.test_nets[0].blobs[ip_softmax].data.argmax(1), solver.test_nets[0].blobs['label'].data
       

def get_batch_from_lmdb(lmdb_cursor, batch_size):
    img_batch = []
    ground_truth_batch = []
    for i in range (batch_size):

        datum = caffe_pb2.Datum()        
        
        lmdb_cursor.next()
        #    print lmdb_cursor.key()
        value = lmdb_cursor.value()
        
        datum.ParseFromString( value )
        
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        img_batch.append(data)
        ground_truth_batch.append(label)
        print "{} | label {}".format(lmdb_cursor.key(), label)
    time.sleep(5)
	
    return img_batch, ground_truth_batch
 

def ensemble_accuracy(ensemble_batch_predictions, ground_truth_batch):
    correct = 0
    for i in range(len(ground_truth_batch)):
        predicted_labels = []
        for matrix_labels in ensemble_batch_predictions:
            predicted_labels.append( matrix_labels[i] ) 
        counts = np.bincount(predicted_labels)
        consensum_choice = np.argmax(counts)
        
        if consensum_choice == ground_truth_batch[i]:
            correct += 1
            
    return correct / 1e2
    

def plot_accuracy(test_acc, test_interval, output_plot_path):
    # Generates plot
    fig, ax2 = subplots()
    ax2.plot(test_interval * range(len(test_acc)), test_acc, 'r')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Test Mean Accuracy: {:.2f}'.format(np.mean(test_acc)))
    # Saves plot
    fig.savefig(output_plot_path)


# Weights of the pretrained networks that compose the ensemble 
ensemble_weights = []  
for i in range(n_nets_in_ensemble):
    ensemble_weights.append(weights.format(i))

niter = 10
test_interval = 1 

test_acc = []
for it in range(n_nets_in_ensemble):
    
    print 'Testing batch it {}...'.format(it)
 
    matrices_labels = []
    for idx, model_weights in enumerate(ensemble_weights):           
        print "Net {} loaded".format(idx)
        solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
        solver = caffe.get_solver(solver_path)
        solver.net.copy_from(model_weights)
#        solver.restore(model_weights)
        solver.step(1)
		
        predictions, ground_truth_batch = test_by_solver(solver, it)
        matrices_labels.append(predictions)
        
    print "Testing ensemble it {}...".format(it)
    
    accuracy = ensemble_accuracy(matrices_labels, ground_truth_batch)
    test_acc.append(accuracy)# When test net's batch size is 100
    
    actual_accuracy = 0
    actual_accuracy += sum(predictions == ground_truth_batch)
                                         
    print "It {}, Ensemble accuracy achieved = {} ; isolated accuracy = {} ; accuracy layer = {}".format(it, accuracy, actual_accuracy/1e2, solver.test_nets[0].blobs['accuracy'].data)

    for i in range(len(matrices_labels)):
        print "solver {} accuracy = {}".format(i, sum(matrices_labels[i] == ground_truth_batch))
        
    time.sleep(10)
    # Reset batches for the next test iteration
    img_batch = []
    ground_truth_batch = []

plot_accuracy(test_acc, test_interval, output_plot_path)



