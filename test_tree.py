#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 18:20:15 2023

@author: ben
"""

import pytest
from DecisionTreeCategorical import DecisionTree
import numpy as np
import pandas as pd

dt = DecisionTree()
singleLabelData = [[0,1],[2,1],[4,1],[7,1]]
singleFeatureData = [[0,2],[0,1],[0,0],[0,3]]

#test single label function works
def test_singleLabel():
    assert dt._singleLabel(singleLabelData) == True
def test_notSingleLabel():
    assert dt._singleLabel(singleFeatureData) == False
    
#test single feature function works
def test_singleFeature():
    assert  dt._singleFeature(singleFeatureData) == True
def test_notSinglefeature():
    assert dt._singleFeature(singleLabelData) == False



#test featureTypes list made correctly
features = np.array(pd.DataFrame([[1,"cat","dog",0.3],
                     [2,"cat","dog",0.2],
                      [3,"cat","dog",0.1],
                       [4,"cat","dog",0.3],
                        [5,"cat","dog",0.3],
                         [6,"cat","dog",0.3],
                          [7,"cat","dog",0.3],
                           [8,"cat","dog",0.3],
                            [9,"cat","dog",0.2],
                             [10,"cat","dog",0.4],
                              [11,"cat","dog",0.5],
                               [12,"cat","dog",0.3],
                                [13,"cat","dog",0.3],
                                 [14,"cat","dog",0.3],
                                  [15,"cat","dog",0.3],
                                   [16,"cat","dog",0.3],
                                    [17,"cat","dog",0.2],
                                     [18,"cat","dog",0.3],
                                      [19,"cat","dog",0.2],
                                       [20,"cat","dog",0.3],
                                        [21,"cat","dog",0.3],
                                         [22,"cat","dog",0.3],
                                          [23,"cat","dog",0.3],
                                           [24,"cat","dog",0.3],
                                            [25,"cat","dog",0.3]]))


# first feature continuous, second and third strings so categorical, and last has less than 20 unique values so also categorical
def test_featureTypes():
    assert dt._get_feature_types(features) == ['continuous','categorical','categorical', 'categorical']
    
    
# test different parameters on iris dataset 
from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']

#test gini calculations
def test_giniMin():
    dt = DecisionTree(criterion='gini')
    data = [[1,1],[2,1],[3,1]]
    assert dt._calculate(data, dt._gini) == 0
    
def test_giniMax():
    dt = DecisionTree(criterion='gini')
    data = [[1,1],[2,2]]
    assert dt._calculate(data, dt._gini) == 0.5
    
def test_giniOneValue():
    dt = DecisionTree(criterion='gini')
    data = [[1,1]]
    assert dt._calculate(data, dt._gini) == 0
    
#test entropy values
def test_entropyMin():
    dt = DecisionTree(criterion='entropy')
    data = [[1,1],[2,1],[3,1]]
    assert dt._calculate(data, dt._entropy) == 0
    
def test_entropyMax():
    dt = DecisionTree(criterion='entropy')
    data = [[1,1],[2,2]]
    assert dt._calculate(data, dt._entropy)  == 1
   
def test_entropyOneValue():
    dt = DecisionTree(criterion='entropy')
    data = [[1,1]]
    assert dt._calculate(data, dt._entropy) == 0

#test max_depth
def test_max_depth0():
    dt = DecisionTree(max_depth=0)
    dt.fit(X,y)
    assert dt.getDepth() == 0
def test_max_depth1():
    dt = DecisionTree(max_depth=1)
    dt.fit(X,y)
    assert dt.getDepth() == 1
def test_max_depth2():
    dt = DecisionTree(max_depth=2)
    dt.fit(X,y)
    assert dt.getDepth() == 2

#test min_samples_split

def test_minSamplesSplit3():
    dt = DecisionTree(min_samples_split=3)
    sortedData = np.array([[1,1],[2,1]])
    labels = np.array([1,1])
    #should make leaf node with a value
    assert dt._build(sortedData, labels).value is not None
    
def test_minSamplesSplit3_2():
    dt = DecisionTree(min_samples_split=3)
    sortedData = np.array([[1,0],[2,4],[3,3]])
    labels = np.array([1,1,0])
    dt.featureTypes = ['continuous','continuous']
    #should split so no value assigned
    assert dt._build(sortedData, labels).value is None
    
def test_minSamplesSplit3_3():
    dt = DecisionTree(min_samples_split=3)
    sortedData = np.array([[1,0,8],[2,4,6],[3,3,2],[4,3,1]])
    labels = np.array([1,1,0,1])
    dt.featureTypes = ['continuous','continuous','continuous']
    #should split so no value assigned
    assert dt._build(sortedData, labels).value is None
   
#test min_samples_leaf
def test_minSamplesLeaf2():
    dt = DecisionTree(min_samples_leaf=2)
    sortedData = np.array([[1,0],[2,1],[3,1]])
    dt.featureTypes = ['continuous']
    assert dt._findAllMidPoints(sortedData, 0) == []
    
def test_minSamplesLeaf2_2():
    dt = DecisionTree(min_samples_leaf=2)
    sortedData = np.array([[1,0],[2,0],[3,1],[4,1]])
    dt.featureTypes = ['continuous']
    # just one possible split 
    assert dt._findAllMidPoints(sortedData, 0) == [(2.5,2)]
    
def test_minSamplesLeaf2Cat():
    dt = DecisionTree(min_samples_leaf=2)
    sortedData = np.array([[1,0],[1,1],[2,1],[2,1],[2,0]])
    dt.featureTypes = ['categorical']
    #return 2 values to split on
    assert dt._findAllMidPoints(sortedData, 0) == [(1,2), (2,3)]

def test_minSamplesLeaf2_2Cat():
    dt = DecisionTree(min_samples_leaf=2)
    sortedData = np.array([[1,0],[2,1],[2,1]])
    dt.featureTypes = ['categorical']
    #return 2 values to split on
    assert dt._findAllMidPoints(sortedData, 0) == []
    
#test memoryUse returns value
def test_memory():
    dt = DecisionTree()
    dt.fit(X, y)
    assert dt.getMemoryUse() > 0
#test trainingTime returns value
def test_time():
    dt = DecisionTree()
    dt.fit(X, y)
    assert dt.getTrainingTime() > 0
    

#test midpoints on continuous data

def test_midpoints():
    sortedData = [[1,1],[2,1],[3,2]]
    dt = DecisionTree()
    dt.featureTypes = ['continuous']
    assert dt._findAllMidPoints(sortedData, 0) == [(1.5, 1), (2.5, 2)]
    

def test_midpointsOneValue():
    sortedData = [[1,1]]
    dt = DecisionTree()
    dt.featureTypes = ['continuous']
    assert dt._findAllMidPoints(sortedData, 0) == []
    
def test_midpointsCategorical():
    sortedData = [["cat",1],['dog',1],['cat',2]]
    dt = DecisionTree()
    dt.featureTypes = ['categorical']
    assert dt._findAllMidPoints(sortedData, 0) == [('cat', 2), ('dog', 1)]
    
    
def test_midpointsCategoricalOneValue():
    sortedData = [["cat",1]]
    dt = DecisionTree()
    dt.featureTypes = ['categorical']
    assert dt._findAllMidPoints(sortedData, 0) == []
    
#test findbestsplit with different data

def test_findBestSplitOneLabel():
    sortedData = [[0,1],[1.5,1]]
    dt = DecisionTree()
    dt.featureTypes = ['continuous']
    assert dt._findBestSplitWithData(sortedData, 0) == False
    
def test_findBestSplitOneFeature():
    sortedData = [[1.5,0],[1.5,1]]
    dt = DecisionTree()
    dt.featureTypes = ['continuous']
    assert dt._findBestSplitWithData(sortedData, 0) == False
    
def test_findBestSplitContinuous():
    sortedData = [[1,0],[1.5,1], [2.0,1], [3.0,0], [3.5,1]]
    dt = DecisionTree()
    dt.featureTypes = ['continuous']
    assert dt._findBestSplitWithData(sortedData, 0)[1] == 1.25
    
    
    