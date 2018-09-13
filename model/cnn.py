

class CNN(object):
    def __init__(self, weights, softmax_layer, label_layer):
        self.weights = weights
        self.softmax_layer = softmax_layer
        self.label_layer = label_layer
