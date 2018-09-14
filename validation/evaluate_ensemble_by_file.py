import caffe
import numpy as np

from contrib.plot_tools import AccuracyPlot
from model.cnn import CNN
from model.ensemble import VotingEnsemble

caffe.set_device(0)
caffe.set_mode_gpu()

tfm_root = '/home/asema/work/PUBLICATION/'
output_plot_path = tfm_root + 'test_plots/ensemble/finalphase/BreaKHis_testing_ensemble_3fcl_novo.jpg'


class ValidateEnsembleByFile(object):

    def compute_ensemble_output(self):
        ensemble_predictions = []
        for n in range(self.steps):
            preds_in_batch = []
            for m in range(self.batch_size):
                single_nets_output = []
                for net, predictions in self.nets_predictions.iteritems():
                    single_nets_output.append(predictions[n][m])

                preds_in_batch.append(self.ensemble.get_consensus(single_nets_output))

            ensemble_predictions.append(preds_in_batch)
        return ensemble_predictions

    def compute_accuracy(self):
        accuracy_list = []
        for n in range(self.steps):
            acc = sum([1 for i, j in zip(self.ensemble_outputs[n], self.ground_truth[n]) if i == j])
            acc /= float(self.batch_size)
            accuracy_list.append(acc)
        return accuracy_list

    def __init__(self, ensemble, steps=10):
        self.steps = steps
        self.ensemble = ensemble
        self.nets_predictions = {}  # It will store the predictions made by each net in the ensemble
        from contrib.file_tools import read_predictions_batch, read_ground_truth_batch, compose_file_name

        self.ground_truth = []
        for it in range(steps):
            gt_path = compose_file_name('groundtruth', 0, it)
            self.ground_truth.append(read_ground_truth_batch(gt_path))

        for cnn in self.ensemble.nets:
            self.nets_predictions[cnn] = []

        for i_net, cnn in enumerate(self.ensemble.nets):
            for it in range(steps):
                preds_path = compose_file_name('preds', i_net, it)
                self.nets_predictions[cnn].append(read_predictions_batch(preds_path))

        self.batch_size = len(self.ground_truth[0])
        self.ensemble_outputs = self.compute_ensemble_output()
        self.accuracy_list = self.compute_accuracy()


def test_ensemble(selected_nets, ensemble_class):

    SOFTMAX_LAYER = 'ip3'
    LABEL_LAYER = 'label'
    STEPS = 10

    # Import caffe, adding it to sys.path if needed. Make sure you've built pycaffe.
    weights = tfm_root + 'test_snapshots/secondphase/BreaKHis_cnn4_debugging-{}_iter_50000.caffemodel'

    ensemble_weights = [CNN(weights.format(i), SOFTMAX_LAYER, LABEL_LAYER)
                        for i, net in enumerate(selected_nets) if selected_nets[i]]

    ensemble = ensemble_class(ensemble_weights)

    test_result = ValidateEnsembleByFile(ensemble, steps=STEPS)

    return test_result


def compute_ensemble_mean_accuracy(selected_nets, ensemble_class):
    test_result = test_ensemble(selected_nets, ensemble_class)
    return np.mean(test_result.accuracy_list)


if __name__ == '__main__':

    """ 
    Example using all the 10 already trained networks
    """
    ensemble_cls = VotingEnsemble
    test_res = test_ensemble([1] * 10, ensemble_cls)
    plot = AccuracyPlot(test_res.accuracy_list, output_plot_path)
