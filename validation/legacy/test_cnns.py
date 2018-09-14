# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:21:38 2018

@author: root
"""

# Import caffe, adding it to sys.path if needed. Make sure you've built pycaffe.
caffe_root = '/home/cristian/caffe-master/'
tfm_root = '/home/cristian/TFM/'
output_plot_path= tfm_root+'test_plots/ensemble/finalphase/BreaKHis_testing_ensemble_3fcl.jpg'
weights = "/home/cristian/TFM/test_snapshots/secondphase/BreaKHis_cnn4_debugging-{}_iter_50000.caffemodel"
solver_path = '/home/cristian/TFM/test_solvers/base_lr_0.0001/debugging-BreaKHis_second_phase_3fcl_type_Adam_solver_0.prototxt'
ip_softmax = 'ip3'
n_nets_in_ensemble = 10


import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from contrib.file_tools import writef
from contrib.file_tools import compose_file_name


#caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
import time
import lmdb

    
def test_in_batch(net, img_batch, ground_truth_batch):
    if len(img_batch) != len(ground_truth_batch):
        raise ValueError('Las dimensiones de los batches no coinciden')

    arch_info = [(k, v.data.shape) for k, v in net.blobs.items()]
    print("Blobs dimensions {}".format(arch_info))
    time.sleep(10)
    for idx in range(len(img_batch)):
        net.blobs['data'].data[idx] = img_batch[idx]
        net.blobs['label'].data[idx] = ground_truth_batch[idx]

    output = net.forward()
    print("output", output)
    time.sleep(5)
    return net.blobs[ip_softmax].data.argmax(1)


def test_by_solver(solver, current_it):
    for i in range((current_it+1)*10):
        solver.test_nets[0].forward()

    predictions = solver.test_nets[0].blobs[ip_softmax].data
    ground_truth_batch = solver.test_nets[0].blobs['label'].data
    return predictions, ground_truth_batch
       

def get_batch_from_lmdb(lmdb_cursor, batch_size):
    img_batch = []
    ground_truth_batch = []
    for i in range (batch_size):

        datum = caffe_pb2.Datum()

        lmdb_cursor.next()
        value = lmdb_cursor.value()

        datum.ParseFromString( value )
        
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        img_batch.append(data)
        ground_truth_batch.append(label)
        print("{} | label {}".format(lmdb_cursor.key(), label))
    time.sleep(5)
    return img_batch, ground_truth_batch
 

def compute_ensemble_accuracy(ensemble_batch_predictions, ground_truth_batch):
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


if __name__ == '__main__':

    lmdb_file = tfm_root+"Trial_Dataset/cancer_test_lmdb/"
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    test_batch_size = 1
    test_interval = test_batch_size
    # Network architecture
    model_def_net = tfm_root + "brasileños/BreaKHis_cnn4_train_test.prototxt"

    # Weights of the pretrained networks that compose the ensemble

    niter = 10
    test_acc = []
    ensemble_weights = []
    for i in range(n_nets_in_ensemble):
        ensemble_weights.append(weights.format(i))

    for it in range(niter):
        labels_matrices = []
        for idx, model_weights in enumerate(ensemble_weights):

            print("Net {} loaded".format(idx))

            solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
            solver = caffe.get_solver(solver_path)
            solver.net.copy_from(model_weights)
            solver.step(1)

            batch_predictions, ground_truth_batch = test_by_solver(solver, it)

            writef(batch_predictions, compose_file_name('preds', idx, it))

            labels_matrices.append(batch_predictions)

        print("Testing ensemble it {}...".format(it))
