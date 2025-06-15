#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import override
from sklearn.mixture import GaussianMixture
from .clustering_base import ClusteringBase

class GMMCluster(ClusteringBase):

    def __init__(self, num_classes, label_offset = 0, random_state=None):
        super().__init__(num_classes, label_offset)

        self.random_state = random_state
        self.model = GaussianMixture(n_components=num_classes, random_state=random_state)

    @override
    def fit(self, features):
        self.model.fit(features)

    @override
    def _pre_predict(self, features):
        return self.model.predict(features)
