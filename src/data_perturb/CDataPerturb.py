import numpy as np
from abc import ABC, abstractmethod


class CDataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        # x should be a 1D array (single sample with multiple features)
        if x.ndim != 1:
            raise ValueError("Input data must be a 1D array representing a single sample.")
        pass

    def perturb_dataset(self, X : np.ndarray):
        """Perturb all samples in the dataset X (shape: n_samples x n_features)"""
        X_perturbed = X.copy()
        for i in range(X_perturbed.shape[0]):
            X_perturbed[i] = self.data_perturbation(X_perturbed[i])
        return X_perturbed