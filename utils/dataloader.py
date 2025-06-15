#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Hao Dai

import numpy as np
from continuum.datasets import InMemoryDataset
from continuum import ClassIncremental

from abc import ABC
from abc import abstractmethod
from typing import override

from collections import namedtuple

class IncrementalLoader(ABC):

    def __init__(self, data_dir, pretrained_model_name, base, increment):
        self.features, self.labels, self.test_features, self.test_labels = self._load_npy(data_dir, pretrained_model_name)
        self._train_dataset, self._test_dataset = self._make_dataset(self.features, self.labels,
                                                                     self.test_features, self.test_labels)

    def _load_npy(self, path, model_name):
        features = np.load(path+f"/{model_name}/features-{model_name}.npy")
        labels = np.load(path+f"/{model_name}/labels-{model_name}.npy")
        test_features = np.load(path+f"/{model_name}/test_features-{model_name}.npy")
        test_labels = np.load(path+f"/{model_name}/test_labels-{model_name}.npy")
        return features, labels, test_features, test_labels

    def _make_dataset(self, features, labels, test_features, test_labels):
        train_dataset = InMemoryDataset(features, labels)
        test_dataset = InMemoryDataset(test_features, test_labels)
        return train_dataset, test_dataset

    def _concatenate_data(self, data1, data2):
        if data1 is None:
            return data2
        else:
            return np.concatenate([data1, data2])

    @abstractmethod
    def train_dataloader(self):
        '''
        Return a DataLoader for training
        '''

    @abstractmethod
    def test_dataloader(self):
        '''
        Return a DataLoader for testing
        '''

class DataPoint:
    def __init__(self, x, y):
        self._x = x
        self._y = y

class ClassIncrementalLoader(IncrementalLoader):
    
    def __init__(self, data_dir, pretrained_model_name,base=50, increment=10):
        super().__init__(data_dir=data_dir, pretrained_model_name=pretrained_model_name, base=base, increment=increment)
        self._increment = increment
        self._base = base

    @override
    def train_dataloader(self):
        self.train_scenario = ClassIncremental(
            self._train_dataset,
            increment=self._increment,
            initial_increment=self._base,
        )
        return self.train_scenario

    @override
    def test_dataloader(self, mode="novel"):

        test_loader = self.test_scenario = ClassIncremental(
            self._test_dataset,
            increment=self._increment,
            initial_increment=self._base,
        )

        if mode == 'novel':
            return test_loader

        test_data_all_per_stage = []
        test_data_old_per_stage = [(np.array([]), np.array([]))]

        tmp_test_data_x = None
        tmp_test_data_y = None
        for stage_i, test_data in enumerate(test_loader):
            tmp_test_data_x = self._concatenate_data(tmp_test_data_x, test_data._x)
            tmp_test_data_y = self._concatenate_data(tmp_test_data_y, test_data._y)

            test_data_all_per_stage.append((tmp_test_data_x, tmp_test_data_y))
            test_data_old_per_stage.append((tmp_test_data_x, tmp_test_data_y))

        if mode == 'all':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_all_per_stage)

        elif mode == 'old':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_old_per_stage)

        else:
            raise ValueError("mode should be 'all' or 'novel'")



class StrictClassInstanceIncrementalLoader(IncrementalLoader):
    '''
    increment both the number of instances and classes, with each increment, the number of classes is fixed, and the number of known instances of each class and that of novel are fixed.
    '''
    def __init__(self, data_dir, pretrained_model_name,base=50, increment=10, num_labeled = 32000, num_novel_per_stage = 1500, num_known_per_stage = 2000):
        super().__init__(data_dir =data_dir, pretrained_model_name=pretrained_model_name, base=base, increment=increment)
        self._increment = increment
        self._base = base
        self.num_labeled = num_labeled
        self.num_novel_per_stage = num_novel_per_stage
        self.num_known_per_stage = num_known_per_stage

        self.cl = ClassIncrementalLoader(data_dir=data_dir, pretrained_model_name= pretrained_model_name, base= base, increment=increment)


    def _resort_data(self, x, y):
        permutation_idx = np.random.permutation(x.shape[0])
        return x[permutation_idx], y[permutation_idx]

    def _sample_known_instance(self, x, y, num_known_per_stage):
        permutation_idx = np.random.permutation(x.shape[0])
        return x[permutation_idx][:num_known_per_stage], y[permutation_idx][:num_known_per_stage]

    def _train_data_mix(self):
        '''
        return [(stage_1_x, stage_1_y), (stage_2_x, stage_2_y), ...]
        '''
        # mix the data, for labels :
        # 0 - num_known : known instances
        labeled_per_class = self.num_labeled // self._base

        num_novel_per_stage_per_class = self.num_novel_per_stage // self._increment

        unlabeled_x_pool = None
        unlabeled_y_pool = None

        dataset_per_stage = []

        train_loader = self.cl.train_dataloader()

        for stage_i, train_data in enumerate(train_loader):
            if stage_i == 0 :
                labeled_x = None
                labeled_y = None

                idx_classes = np.unique(train_data._y)
                for idx in idx_classes:
   
                    labeled_x = self._concatenate_data(labeled_x, train_data._x[train_data._y == idx][:labeled_per_class])
 
                    labeled_y = self._concatenate_data(labeled_y, np.ones(labeled_per_class) * idx)

                    unlabeled_x = train_data._x[train_data._y == idx][labeled_per_class:]
                    unlabeled_y = np.ones(len(unlabeled_x)) * idx

                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                labeled_x, labeled_y = self._resort_data(labeled_x, labeled_y)
                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)

                dataset_per_stage.append((labeled_x, labeled_y))

            else:
                # sample known instance
                known_x, known_y = self._sample_known_instance(unlabeled_x_pool, unlabeled_y_pool, self.num_known_per_stage)

                # sample novel instance
                idx_classes = np.unique(train_data._y)
                unlabeled_novels_x = None
                unlabeled_novels_y = None
                for idx in idx_classes:
                    unlabeled_novels_x =  self._concatenate_data(unlabeled_novels_x, train_data._x[train_data._y == idx][: num_novel_per_stage_per_class])
                    unlabeled_novels_y = self._concatenate_data(unlabeled_novels_y, np.ones(num_novel_per_stage_per_class) * idx)

                    unlabeled_x = train_data._x[train_data._y == idx][num_novel_per_stage_per_class:]
                    unlabeled_y = np.ones(len(unlabeled_x)) * idx

                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                stage_x = self._concatenate_data(known_x, unlabeled_novels_x)
                stage_y = self._concatenate_data(known_y, unlabeled_novels_y)

                stage_x, stage_y = self._resort_data(stage_x, stage_y)

                dataset_per_stage.append((stage_x, stage_y))

                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)

        return dataset_per_stage


                    
    @override
    def train_dataloader(self):
        train_data_per_stage = self._train_data_mix()
        return map(lambda x: DataPoint(x[0], x[1]),train_data_per_stage)


        
    @override
    def test_dataloader(self, mode='all'):
        '''
        mode : str, 'all', 'old' or 'novel'
        '''
        test_loader = self.cl.test_dataloader()

        if mode == 'novel':
            return test_loader

        test_data_all_per_stage = []
        test_data_old_per_stage = [(np.array([]), np.array([]))]

        tmp_test_data_x = None
        tmp_test_data_y = None
        for stage_i, test_data in enumerate(test_loader):
            tmp_test_data_x = self._concatenate_data(tmp_test_data_x, test_data._x)
            tmp_test_data_y = self._concatenate_data(tmp_test_data_y, test_data._y)

            test_data_all_per_stage.append((tmp_test_data_x, tmp_test_data_y))
            test_data_old_per_stage.append((tmp_test_data_x, tmp_test_data_y))

        if mode == 'all':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_all_per_stage)

        elif mode == 'old':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_old_per_stage)

        else:
            raise ValueError("mode should be 'all' or 'novel'")


class StrictPerClassIncrementalLoader(IncrementalLoader):
    '''
    increment both the number of instances and classes, with each increment, the number of classes is fixed, and the number of known instances of each class and that of novel are fixed.
    '''
    def __init__(self, data_dir, pretrained_model_name,base, increment, num_labeled, num_novel_inc, num_known_inc):
        super().__init__(data_dir =data_dir, pretrained_model_name=pretrained_model_name, base=base, increment=increment)
        self._increment = increment
        self._base = base
        self.num_labeled = num_labeled
        self.num_novel_inc = num_novel_inc
        self.num_known_inc = num_known_inc

        self.cl = ClassIncrementalLoader(data_dir=data_dir, pretrained_model_name= pretrained_model_name, base= base, increment=increment)


    def _resort_data(self, x, y):
        permutation_idx = np.random.permutation(x.shape[0])
        return x[permutation_idx], y[permutation_idx]

    def _sample_known_instance(self, x, y, seen_classes, num_known_inc):
        permutation_idx = np.random.permutation(x.shape[0])
        x = x[permutation_idx]
        y = y[permutation_idx]
        remained_x = []
        remained_y = []
        seen_samples_x = []
        seen_samples_y = []
        for c in seen_classes:
            cc_x = x[y == c]
            cc_y = y[y == c]
            seen_samples_x.append(cc_x[:num_known_inc])
            seen_samples_y.append(cc_y[:num_known_inc])
            remained_x.append(cc_x[num_known_inc:])
            remained_y.append(cc_y[num_known_inc:])

        return x, y , np.concatenate(seen_samples_x), np.concatenate(seen_samples_y)

    def _train_data_mix(self):
        '''
        return [(stage_1_x, stage_1_y), (stage_2_x, stage_2_y), ...]
        '''
        # mix the data, for labels :
        # 0 - num_known : known instances
        labeled_per_class = self.num_labeled // self._base

        num_novel_per_stage_per_class = self.num_novel_inc 

        unlabeled_x_pool = None
        unlabeled_y_pool = None

        dataset_per_stage = []

        train_loader = self.cl.train_dataloader()

        seen_classes = None

        for stage_i, train_data in enumerate(train_loader):
            if stage_i == 0 :
                labeled_x = None
                labeled_y = None

                idx_classes = np.unique(train_data._y)

                seen_classes = idx_classes 

                for idx in idx_classes:
   
                    labeled_x = self._concatenate_data(labeled_x, train_data._x[train_data._y == idx][:labeled_per_class])
 
                    labeled_y = self._concatenate_data(labeled_y, np.ones(labeled_per_class) * idx)

                    unlabeled_x = train_data._x[train_data._y == idx][labeled_per_class:]
                    unlabeled_y = np.ones(len(unlabeled_x)) * idx

                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                labeled_x, labeled_y = self._resort_data(labeled_x, labeled_y)
                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)

                dataset_per_stage.append((labeled_x, labeled_y))

            else:
                # sample known instance
                unlabeled_x_pool, unlabeled_y_pool, known_x, known_y = self._sample_known_instance(unlabeled_x_pool, unlabeled_y_pool,  seen_classes, self.num_known_inc)

                # sample novel instance
                idx_classes = np.unique(train_data._y)
                unlabeled_novels_x = None
                unlabeled_novels_y = None

                seen_classes = self._concatenate_data(seen_classes, idx_classes)

                for idx in idx_classes:
                    unlabeled_novels_x =  self._concatenate_data(unlabeled_novels_x, train_data._x[train_data._y == idx][: num_novel_per_stage_per_class])
                    unlabeled_novels_y = self._concatenate_data(unlabeled_novels_y, np.ones(num_novel_per_stage_per_class) * idx)

                    unlabeled_x = train_data._x[train_data._y == idx][num_novel_per_stage_per_class:]
                    unlabeled_y = np.ones(len(unlabeled_x)) * idx

                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                stage_x = self._concatenate_data(known_x, unlabeled_novels_x)
                stage_y = self._concatenate_data(known_y, unlabeled_novels_y)

                stage_x, stage_y = self._resort_data(stage_x, stage_y)

                dataset_per_stage.append((stage_x, stage_y))

                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)

        return dataset_per_stage


                    
    @override
    def train_dataloader(self):
        train_data_per_stage = self._train_data_mix()
        return map(lambda x: DataPoint(x[0], x[1]),train_data_per_stage)


        
    @override
    def test_dataloader(self, mode='all'):
        '''
        mode : str, 'all', 'old' or 'novel'
        '''
        test_loader = self.cl.test_dataloader()

        if mode == 'novel':
            return test_loader

        test_data_all_per_stage = []
        test_data_old_per_stage = [(np.array([]), np.array([]))]

        tmp_test_data_x = None
        tmp_test_data_y = None
        for stage_i, test_data in enumerate(test_loader):
            tmp_test_data_x = self._concatenate_data(tmp_test_data_x, test_data._x)
            tmp_test_data_y = self._concatenate_data(tmp_test_data_y, test_data._y)

            test_data_all_per_stage.append((tmp_test_data_x, tmp_test_data_y))
            test_data_old_per_stage.append((tmp_test_data_x, tmp_test_data_y))

        if mode == 'all':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_all_per_stage)

        elif mode == 'old':
            return map(lambda x: DataPoint(x[0], x[1]), test_data_old_per_stage)

        else:
            raise ValueError("mode should be 'all' or 'novel'")

        
