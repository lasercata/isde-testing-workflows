import numpy as np
from abc import ABC, abstractmethod


class CDataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        if x.shape[1] != 1:
            raise ValueError("Input data must be a 2D array with a single feature column.")
        pass

    def perturb_dataset(self, X : np.ndarray):
        for i in range(X.shape[0]):
            X[i] = self.data_perturbation(X[i])
        return X