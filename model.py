from abc import ABC, abstractmethod
import numpy as np

'''
Base class for all CNN models
'''
class CNN_Model(ABC):
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def build(self):
        pass

    def count_params(self, model):
        non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
        trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
        return { 'trainable_params': trainable_params, 'non_trainable_params': non_trainable_params }
