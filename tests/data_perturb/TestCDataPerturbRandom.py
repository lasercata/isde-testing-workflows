#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##-Imports
import unittest
import numpy as np

from src.data_perturb.CDataPerturbRandom import CDataPerturbRandom

##-Tests
class TestCDataPerturbRandom(unittest.TestCase):

    def setUp(self):
        n_samples = 100
        n_features = 20

        self.test_ndarr = np.zeros(shape=(n_samples,))
        # self.test_ndarr = np.zeros(shape=(n_samples, n_features))

    def test_data_perturbation(self):
        for k in (0, 1, 3, 100):
            p = CDataPerturbRandom(K=k)
            res = p.data_perturbation(self.test_ndarr)
            
            nb_changed = len([i for i in self.test_ndarr if i != 0]) # Count the number of changed values
            self.assertLessEqual(nb_changed, k, f'More changes than {k}!')
