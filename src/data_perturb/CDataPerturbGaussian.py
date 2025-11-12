import numpy as np
from src.data_perturb.CDataPerturb import CDataPerturb


class CDataPerturbGaussian(CDataPerturb):
    def __init__(self, min = 0, max = 255, sig = 100.0):
        self._min_value = min
        self._max_value = max
        self._sigma = sig

    @property
    def min_value(self):
        return self._min_value
    
    @property
    def max_value(self):
        return self._max_value
    
    @property
    def sigma(self):
        return self._sigma
    
    @min_value.setter
    def min_value(self, value):
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        self._max_value = value

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_perturbation(self, data : np.ndarray) -> np.ndarray:
        super().data_perturbation(data)
        for i in range(data.shape[0]):
            noise = self.sigma * np.random.randn()
            data[i] = np.clip(data[i] + noise, self.min_value, self.max_value)
        return data