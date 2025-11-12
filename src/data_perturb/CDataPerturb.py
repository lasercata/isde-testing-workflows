import numpy as np
from abc import ABC, abstractmethod


class CDataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        pass

    def perturb_dataset(self, X : np.ndarray):
        for i in range(X.shape[0]):
            X[i] = self.data_perturbation(X[i])
        return X