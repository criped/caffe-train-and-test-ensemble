
from abc import ABCMeta, abstractmethod
import numpy as np


class Ensemble(object):

    __metaclass__ = ABCMeta    
    
    def __init__(self, nets):
        self.nets = nets

    @abstractmethod
    def get_consensus(self, predictions):
        """
        To be implemented by child 
        """
        pass
        

class VotingEnsemble(Ensemble):
    def __init__(self, nets):
        super(VotingEnsemble, self).__init__(nets)
    
    def get_consensus(self, predictions): 
        """
        Consensum implementation is the voting value
        """
        choices = [c_scores.index(max(c_scores)) for c_scores in predictions]
    
        counts = np.bincount(choices)
        return np.argmax(counts)
  

class MaxEnsemble(Ensemble):
    def __init__(self, nets):
        super(MaxEnsemble, self).__init__(nets)
    
    def get_consensus(self, predictions): 
        """
        Consensum implementation is the max value
        """

        max_class_scores = [max(x) for x in zip(*predictions)]
        consensus = max_class_scores.index(max(max_class_scores))
        return consensus


class MeanEnsemble(Ensemble):
    def __init__(self, nets):
        super(MeanEnsemble, self).__init__(nets)
    
    def get_consensus(self, predictions): 
        """
        Consensum implementation is the mean value
        """

        mean_class_scores = [sum(x)/float(len(x)) for x in zip(*predictions)]
            
        consensus = mean_class_scores.index(max(mean_class_scores))
        return consensus
        

class MedianEnsemble(Ensemble):
    def __init__(self, nets):
        super(MedianEnsemble, self).__init__(nets)
    
    def get_consensus(self, predictions): 
        """
        Consensum implementation is the median value
        """

        mean_class_scores = [np.median(x) for x in zip(*predictions)]
            
        consensus = mean_class_scores.index(max(mean_class_scores))
        return consensus

   
Ensemble.register(VotingEnsemble)
Ensemble.register(MaxEnsemble)
Ensemble.register(MeanEnsemble)
Ensemble.register(MedianEnsemble)
