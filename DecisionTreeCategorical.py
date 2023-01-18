#Decision Tree Classifier 
from operator import itemgetter # We use this import for sorting
import numpy as np
import time 
import tracemalloc


class Node:
    '''
    Holds information on node of a tree.
    
    :param feature: feature index
    :param gain: gini index or info gain
    :param splitValue: the value the tree is split  on at this node
    :param left: left branch of tree
    :param right: right branch of tree
    :param value: value assigned (for leaf nodes)
    '''
    def __init__(self, feature=None, gain=None, splitValue=None, left=None, right=None, value=None):
        # Constructor
        self.feature = feature
        self.gain = gain
        self.splitValue = splitValue
        self.left = left
        self.right = right 
        self.value = value
        
        
        

class DecisionTree :
    '''
    Decision Tree classifier.
    
    :param min_samples_split: minimum samples needed in a node to split on
    :param min_samples_leaf: minimum samples needed in a leaf node
    :param max_depth: maximum depth of tree
    :param criterion: the criterion for splitting data (gini index or entropy)
    '''
    def __init__ ( self , min_samples_split=2, min_samples_leaf=1, max_depth=float('inf'), criterion='gini') :
        # Constructor
        self.root =None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.trainingTime = None
        self.depth = None
        self.memoryUse = None
        self.featureTypes = None
    
    '''
    An auxiliary method to find all midpoints between the feature , 
    with the corresponding position in the sortedData list where the split should happen.
    
    :param sortedData: ordered data of feature and corresponding labels
    :param ftIdx: feature index
    :return: list of all midpoints 
    '''
    def _findAllMidPoints ( self , sortedData, ftIdx) :
        listOfMidPoints = [] # Start with an empty list , and add the split points one by one
        splitPosition = 0
        if self.featureTypes[ftIdx] == 'continuous':
            for n in range (len ( sortedData ) -1) :
                splitPosition += 1
                if ( sortedData [ n ][0] == sortedData [ n +1][0]) : # We need to watch out for repeated items
                    continue
                # Check min_samples_leaf criteria is true and we can append the potential midpoint
                if (splitPosition < self.min_samples_leaf) or ((len(sortedData)-splitPosition) < self.min_samples_leaf):
                    continue
                listOfMidPoints . append ((( sortedData [ n ][0] + sortedData [ n +1][0]) /2 , splitPosition ) ) 
                # A tuple representation , to store the splitting value , 
                # and also the position in the list that would correspond to that split .
        if self.featureTypes[ftIdx] == 'categorical':
                
            uniqueValues = np.unique(np.array(sortedData)[:,0])
            sortedDataTemp = np.array(sortedData)
            for n in range(len(uniqueValues)):
                value = uniqueValues[n]
                left = sortedDataTemp[sortedDataTemp[:,0]==value]
                if (left.shape[0] < self.min_samples_leaf) or ((sortedDataTemp.shape[0]-left.shape[0]) < self.min_samples_leaf):
                    continue
                listOfMidPoints.append((value, len(left)))
                
        return listOfMidPoints
    
    '''
    Calculates criterion/gain based on function (entropy or gini).
    
    :param data: the labels data
    :param func: criterion/gain function
    :return: criterion/gain for the data
    '''
    def _calculate(self, data, func):
        uniqueLabels = np.unique(np.array(data)[:,1])
        labelCount = [0] * len(uniqueLabels)
        
        for item in data:
            for i in range(len(uniqueLabels)):
                if item[1] == uniqueLabels[i]:
                    labelCount[i] = labelCount[i] + 1
        
        gain = 0
        for num in labelCount:
            gain += func(num, len(data))
        
        if func == self._gini:
            return 1-gain
        if func == self._entropy:
            return gain
            
    '''
    Calculates entropy.
    
    :param labelCount: the number of labels
    :param length: length of dataset
    :return: entropy
    '''
    def _entropy(self, labelCount, length):
        return -((labelCount/length) * np.log2(labelCount/length))
    
    '''
    Calculates the " gini purity " of a dataset.
    
    :param labelCount: the number of labels
    :param length: length of dataset
    :return: gini index
    '''
    def _gini(self, labelCount, length):
        return (labelCount/length)**2
            
        
    
    '''
    Auxiliary method to return a left and right dataset , with its corresponding gini/gain value , given a certain split point .
    
    :param splitPoint: point to split on in data
    :param splitValue: value at splitPoint
    :param sortedData: data to split on in order
    :param ftIdx: feature index
    :return: tuple of gain/gini, splitValue, left-side dataset, right-side dataset
    '''
    def _split ( self , splitPoint , splitValue , sortedData, ftIdx) :
        
        # Left and right dataset split in different ways depending on continuous or categirical data
        if self.featureTypes[ftIdx] == 'continuous':
            # Dividing the dataset into left and right datasets
            leftSide = sortedData[0: splitPoint]
            rightSide = sortedData[splitPoint :]
        if self.featureTypes[ftIdx]== 'categorical':
            sortedDataTemp = np.array(sortedData)
            leftSide = sortedDataTemp[sortedDataTemp[:,0] == splitValue].tolist()
            rightSide = sortedDataTemp[sortedDataTemp[:,0] != splitValue].tolist()
        # Gini calculation
        if self.criterion == 'gini':
            # Calculate the Gini index for each 
            leftGini = self . _calculate ( leftSide, self._gini)
            rightGini = self . _calculate ( rightSide, self._gini)
            
            
            # Returns a weighted summation of the left and right gini , based on the size of those datasets in relation to the full dataset .
            giniValue = (len( leftSide ) /len( sortedData ) ) * leftGini + (len( rightSide ) /len( sortedData ) ) * rightGini
            return ( round ( giniValue ,4) , splitValue , leftSide , rightSide )
        
        # Information gain calculation
        elif self.criterion == 'entropy':
            # Calculate info gain for each
            leftEntropy = self._calculate(leftSide, self._entropy)
            rightEntropy = self._calculate(rightSide, self._entropy)
            parentEntropy = self._calculate(sortedData, self._entropy)
            
            leftProp = len(leftSide)/len(sortedData)
            rightProp = len(rightSide)/len(sortedData)
            
            #made negative as algorithm is trying to minimise 
            infoGain = -(parentEntropy - (leftProp*leftEntropy + rightProp*rightEntropy))
            return (round( infoGain ,4) , splitValue , leftSide , rightSide )
        
    '''
    Method called to find best split on dateset.
    
    :param X: features data
    :param y: labels
    :return: the best split of the data based of criterion chosen
    '''   
    def _findBestSplit (self, X, y) :
        #loop through each feature and find best split
        idx = 0
        #set min to be max possible
        minGain = float("inf")
        for ftIdx in range(X.shape[1]):
            feature = X[:,ftIdx]
            sortedData = sorted(zip(feature,y), key=itemgetter(0))
            best = self._findBestSplitWithData(sortedData, ftIdx)
            #check if edge cases fail so leaf node needs to be made
            if best is False:
                #gain is false when leaf node
                return {self.criterion:False}
                
            elif best[0] < minGain:
                minGain = best[0]
                bestSplit = best
                idx=ftIdx
                
        #combine data all back together sorted on best feature
        bestSortedData = np.array(sorted(np.concatenate((X,y[:, None]),axis=1),key=itemgetter(idx)))
        
        #different split conditions for types of feature
        if self.featureTypes[idx] == 'continuous':
            #split on split point
            best = {
                'ftIdx': idx,
                self.criterion: bestSplit[0],
                'splitValue': bestSplit[1],
                'left': bestSortedData[:len(bestSplit[2])],
                'right': bestSortedData[len(bestSplit[2]):]
            }
        if self.featureTypes[idx] == 'categorical':
            #split based on specific feature value 
            dataType = type(bestSortedData[:,idx][0])
            if dataType is int:
                splitValue = int(bestSplit[1])
            else:
                splitValue = bestSplit[1]
            best = {
                'ftIdx': idx,
                self.criterion: bestSplit[0],
                'splitValue': splitValue,
                'left': bestSortedData[bestSortedData[:,idx] == splitValue],
                'right': bestSortedData[bestSortedData[:,idx] != splitValue]
            }
        return best
    
    '''
    Finds best split with data.
    
    :param sortedData: ordered data, feature and labels
    :param ftIdx: feature index
    :return: best split from midpoints
    '''
    def _findBestSplitWithData (self, sortedData, ftIdx) :
        # Step 0: edge cases :
        ## If there is a single label or a single feature value , return False , we don ’t have to split any further
        if ( self._singleLabel(sortedData)):
            return False
        if ( self._singleFeature(sortedData)):
            return False
        
        # Step 1: find all midPoints :
        
        listOfMidpoints = self._findAllMidPoints(sortedData, ftIdx)
        
        # Step 2: calculate split and gini index
        listOfPotentialSplits = []
        
        for n in range(len(listOfMidpoints)):
            split = self._split(listOfMidpoints[n][1], listOfMidpoints[n][0], sortedData, ftIdx)
            listOfPotentialSplits.append(split)
        
        # Step 3: find the solution with minimum gini index/ max gain
        
        minimum = float("inf")
        minimumPos = 0
        
        for n in range(len(listOfPotentialSplits)):
            if listOfPotentialSplits [ n ][0] < minimum:
                minimum = listOfPotentialSplits[n][0]
                minimumPos = n
        
         ## Check for duplicates :
        # for n in range (len ( listOfPotentialSplits ) ) :
        #     if ( listOfPotentialSplits [ n ][0] == minimumGini ) and ( n != minimumGiniPos ) :
        #         print (" Duplicated Solution :")
        #         print ( listOfPotentialSplits [ n ])
        
        #if listOfPotentialSplits is empty then there are no potential splits due to failure of min_samples_leaf criteria
        #return false so leaf node made instead
        if listOfPotentialSplits == []:
            return False
        return listOfPotentialSplits[minimumPos]
    
    '''
    Checks if there only items of one of the labels . In that case, there are no further splits.
    
    :param sortedData: ordered data, feature and labels
    :return: true if more than 1 label
    '''
    def _singleLabel(self, sortedData):
        firstLabel = sortedData[0][1]
        
        for item in sortedData :
            if ( item [1] != firstLabel ) :
                return False
        
        return True
    
    '''
    Checks if all items have the same feature , as in that case a split cannot be created .
    
    :param sortedData: ordered data, feature and labels
    :return: true if more than 1 feature
    '''
    def _singleFeature ( self , sortedData ) :
        firstFeature = sortedData [0][0]
        
        for item in sortedData :
           if ( item [0] != firstFeature ) :
               return False
        
        return True
    
    '''
    Auxillary method that recursively builds decision tree.
    
    :param X: features data
    :param y: labels 
    :param depth: depth of tree at that point, initially 0
    :return: Node of each part of the tree (all connected)
    '''
    def _build(self, X, y, depth=0):
        n_rows, n_cols  = X.shape
        
        #check if need to stop iterations and make leaf node
        if (n_rows >= self.min_samples_split) and (depth < self.max_depth):
            #recursively call build on right and left sides to build tree   
            bestSplit = self._findBestSplit(X,y)

            #check if best split is not leaf node
            if bestSplit[self.criterion] is not False:
                #build tree on left
                left = self._build(bestSplit['left'][:,:-1], bestSplit['left'][:,-1], depth=depth+1)
                #take data from left side of split and call build again - finds the best split 
                
                #take data from right side of split and build again
                right = self._build(bestSplit['right'][:,:-1], bestSplit['right'][:,-1], depth=depth+1)
                #return node with best_split variables
                return Node(feature=bestSplit['ftIdx'], gain=bestSplit[self.criterion], splitValue=bestSplit['splitValue'], left=left, right=right)
        
        self.depth = depth
        # Leaf node - value is the most common target value 
        return Node(value=max(set(y.tolist()), key = y.tolist().count))
    
    '''
    Intermediate function that returns the data types of the features at the beginning when training.
    
    :param X: data
    :return: data types of features'''
    def _get_feature_types(self, X):
        n_rows, n_cols  = X.shape
        featureTypes = []
        # Go through features
        for ftIdx in range(n_cols):
            uniqueValues = np.unique(X[:,ftIdx])
            # Conditions for categorical feature.
            if isinstance(uniqueValues[0], str) or len(uniqueValues) <=20:
                featureTypes.append('categorical')
            else:
                featureTypes.append('continuous')
                
        return featureTypes
    
    '''
    Method to fit data to build decision tree.
    
    :param X: features data
    :param y: labels
    '''   
    def fit(self, X, y):
        # Start time and memory
        start = time.time()
        tracemalloc.start()
        
        #check any categorical features, defined as strings or if number of unique values <= 20
        self.featureTypes = self._get_feature_types(X)
        # Call _build which recursively grows tree
        # Assign this to decision tree root
        self.root = self._build(X, y)
        
        end = time.time()
        self.trainingTime = end-start 
        # Memory
        self.memoryUse = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak() 
        # stopping the memory library
        tracemalloc.stop()
    
    '''
    Gets time to train.
    
    :return: training time
    '''
    def getTrainingTime(self):
        return self.trainingTime
    
    '''
    Gets memory use of fit().
    
    :return: memory use
    '''
    def getMemoryUse(self):
        return self.memoryUse
    
    '''
    Gets depth of tree.
    
    :return: depth of tree
    '''
    def getDepth(self):
        return self.depth
    
    '''
    Predicts labels for one row of data.
    
    :param x: one row of feature data
    :param tree: tree to traverse
    :return: the label predicted
    '''
    def _predict(self, x, tree):
        # Tree traversal
        # When reach leaf node, value will be not empty and so return that as label
        if tree.value is not None:
            return tree.value 
        
        ftVal = x[tree.feature]
        
        if self.featureTypes[tree.feature] == 'continuous':
            # Go down left branch
            if ftVal <= tree.splitValue:
                return self._predict(x, tree.left)
            # Go down right branch
            if ftVal > tree.splitValue:
                return self._predict(x, tree.right)
            
        if self.featureTypes[tree.feature] == 'categorical':
            if ftVal == tree.splitValue:
                return self._predict(x, tree.left)
            # Go down right branch
            if ftVal != tree.splitValue:
                return self._predict(x, tree.right)
    '''
    Predicts labels for all data.
    
    :param X: features data
    :return: predicted labels
    '''
    def predict(self, X):
        labels = []
        for x in X:
            labels.append(self._predict(x,self.root))
            
        return labels
    
    '''
    Calculates accuracy score of predictions vs actual labels.
    
    :param X_test: the feature data to be predicted
    :param y_test: the actual labels
    :return: accuracy score
    '''
    def score(self, X_test, y_test):
        # Predict labels
        preds = self.predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if preds[i] == y_test[i]:
                correct +=1
        return correct/len(y_test)
    '''
    Prints the tree. 
    
    :param tree: tree st that point in traversal
    :param indent: the indentation size to be printed
    '''
    def print_tree(self, tree=None, indent="  "):
        # Start condition
        if tree is None:
            tree = self.root 
        # Leaf node means print value
        if tree.value is not None:
            print(tree.value)
        # Traverse tree recursively and print relevant parts 
        else:
            if self.featureTypes[tree.feature] == 'categorical':
                print("Ft_" + str(tree.feature) + " == " + str(tree.splitValue) + ", " + self.criterion + ": " + str(abs(tree.gain)))
            if self.featureTypes[tree.feature] == 'continuous':
                print("Ft_" + str(tree.feature) + " <= " + str(tree.splitValue) + ", " + self.criterion + ": " + str(abs(tree.gain)))
            print(indent + "left: ", end="")
            self.print_tree(tree.left, indent + "  ")
            print(indent + "right: ", end="")
            self.print_tree(tree.right, indent + "  ")
            
            


    
    
    
    
          