import numpy as np
from src.data_perturb.CDataPerturb import CDataPerturb

class CDataPerturbRandom(CDataPerturb):
  def __init__(self, min_value=0, max_value=255, K=100):
    self._min_value = min_value
    self._max_value = max_value
    self._K = K

  @property
  def min_value(self):
    return self._min_value
  
  @min_value.setter
  def min_value(self, value):
    self._min_value = value

  @property
  def max_value(self):
    return self._max_value
  
  @max_value.setter
  def max_value(self, value):
    self._max_value = value

  @property
  def K(self):
    return self._K
  
  @K.setter
  def K(self, value):
    self._K = int(value)
  
  def data_perturbation(self, x):
    if self._K > 0:
      indices = np.random.choice(x.size, min((self._K, x.size)), replace=False)
      x[indices] = np.random.uniform(self._min_value, self._max_value, indices.size)
    return x