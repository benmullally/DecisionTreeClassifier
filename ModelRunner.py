'''
File used to generate CSV files used in report.
Paramater ranges and test_size's ranged.
Print_tree commands also used here.'
'''

import pandas as pd
import numpy as np
import time 
import tracemalloc



from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from DecisionTreeCategorical import DecisionTree  

# Function that makes k-fold's for t-test accuracy scores.
def KFold(X, y, n_splits=10, random_state=42):
    #set random seed
    np.random.seed(random_state)
    #randomly shuffle data
    data = np.concatenate((X,y[:, None]),axis=1)
    np.random.shuffle(data)
    #take n splits
    size = int(len(data)/n_splits)
    row = 0
    
    newXstest = []
    newystest = []
    newXstrain = []
    newystrain = []
    #split data
    for i in range(n_splits):
        newXtest = data[row:(row+size),:-1]
        newytest = data[row:(row+size),-1]
        tempData = np.delete(data,np.s_[row:(row+size)],0)
        newXtrain = tempData[:,:-1]
        newytrain = tempData[:,-1]
        row += size
        #append to list
        newXstest.append(newXtest)
        newystest.append(newytest)
        newXstrain.append(newXtrain)
        newystrain.append(newytrain)
        
    return newXstrain, newystrain, newXstest, newystest
        
#t-test csv file creation
iris = load_iris()
X = iris['data']
y = iris['target']


#test difference between sklearn and original for gini

Xtrain, ytrain, Xtest, ytest = KFold(X, y, n_splits=10, random_state=42)
originalModelAcc = []
sklearnModelAcc = []

for i in range(10):
    original = DecisionTree()
    original.fit(Xtrain[i],ytrain[i])
    originalModelAcc.append(original.score(Xtest[i],ytest[i]))
    
    sk = DecisionTreeClassifier()
    sk.fit(Xtrain[i],ytrain[i])
    sklearnModelAcc.append(sk.score(Xtest[i],ytest[i]))
    
originalModelAcc = pd.DataFrame(originalModelAcc)
originalModelAcc.to_csv("originalModelAcc.csv")
sklearnModelAcc = pd.DataFrame(sklearnModelAcc)
sklearnModelAcc.to_csv("sklearnModelAcc.csv")

#test difference between sklearn and original for entropy
originalModelEntAcc = []
sklearnModelEntAcc = []
for i in range(10):
    original = DecisionTree(criterion='entropy')
    original.fit(Xtrain[i],ytrain[i])
    originalModelEntAcc.append(original.score(Xtest[i],ytest[i]))
    
    sk = DecisionTreeClassifier(criterion='entropy')
    sk.fit(Xtrain[i],ytrain[i])
    sklearnModelEntAcc.append(sk.score(Xtest[i],ytest[i]))
    
originalModelEntAcc = pd.DataFrame(originalModelEntAcc)
originalModelEntAcc.to_csv("originalModelEntAcc.csv")
sklearnModelEntAcc = pd.DataFrame(sklearnModelEntAcc)
sklearnModelEntAcc.to_csv("sklearnModelEntAcc.csv")




#builds models for different paramater ranges
#returns results for both models as pandas dataframes
       
def modelBuilder(criterions, depths, min_leaves, min_splits, X_train, X_test, y_train, y_test):
    results = pd.DataFrame(columns=["criterion", "max_depth", "min_samples_leaf", "min_samples_split", "Accuracy", "Precision", "Recall", "F1", "Time", "Memory","Depth"])
    library_results = pd.DataFrame(columns=["criterion", "max_depth", "min_samples_leaf", "min_samples_split", "Accuracy", "Precision", "Recall", "F1", "Time", "Memory", "Depth"])
    for criterion in criterions:
        for depth in depths:
            for min_leaf in min_leaves:
                for min_split in min_splits:
                    model = DecisionTree(criterion=criterion, max_depth=depth, min_samples_leaf=min_leaf, min_samples_split=min_split)
                    model.fit(X_train, y_train)
                    labels = model.predict(X_test)
                    row = [criterion, depth, min_leaf, min_split, model.score(X_test,y_test), precision_score(y_test, labels, average='macro'), recall_score(y_test, labels, average='macro'), f1_score(y_test, labels, average='macro'), model.getTrainingTime(), model.getMemoryUse(), model.getDepth()]
                    results.loc[len(results)] = row
                    skModel = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf=min_leaf, min_samples_split=min_split)
                    start = time.time()
                    tracemalloc.start()
                    skModel.fit(X_train, y_train)
                    skMem = tracemalloc.get_traced_memory()[1]
                    tracemalloc.reset_peak()
                    tracemalloc.stop()
                    end = time.time()
                    skTime = end-start
                    
                    skLabels = skModel.predict(X_test)
                    
                    skRow = [criterion, depth, min_leaf, min_split, skModel.score(X_test,y_test), precision_score(y_test, skLabels, average='macro'), recall_score(y_test, skLabels, average='macro'), f1_score(y_test, skLabels, average='macro'), skTime, skMem, skModel.get_depth()]
                    library_results.loc[len(library_results)] = skRow
                    
    return (results, library_results)              


#combines all data frames from each test split together
#returns whole dataframe combined
#can then save to csv files
def splitter(X, y, test_sizes):
    
    resultsFrames = []
    library_resultsFrames = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        criterions = ["gini", "entropy"]
        depths = [1,2,3,4,5]
        min_leaves = [1,10,20,50,100]
        min_splits = [2,10,20,50,100]
        resultsDF, library_resultsDF = modelBuilder(criterions, depths, min_leaves, min_splits, X_train, X_test, y_train, y_test)
        size = [test_size] * len(resultsDF)
        resultsDF["test_size"] = size
        library_resultsDF["test_size"] = size
        
        resultsFrames.append(resultsDF)
        library_resultsFrames.append(library_resultsDF)
    
    resultsDF = pd.concat(resultsFrames)
    library_resultsDF = pd.concat(library_resultsFrames)
    return (resultsDF, library_resultsDF)
        

  
tracemalloc.start()
tracemalloc.reset_peak() 

     
# Iris dataset - small and easy classification.
# Should give quick run times and high accuracy
iris = load_iris()
X = iris['data']
y = iris['target']


# Start with unrestricted trees.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("gini: \n")
unrestrictedModel = DecisionTree()
unrestrictedModel.fit(X_train, y_train)
unrestrictedModel.print_tree()
print(unrestrictedModel.score(X_test, y_test))
print(unrestrictedModel.getTrainingTime())
print(unrestrictedModel.getMemoryUse())

skUnrestrictedModel = DecisionTreeClassifier()
start = time.time()
tracemalloc.start()
skUnrestrictedModel.fit(X_train, y_train)
print(tracemalloc.get_traced_memory()[1])
end = time.time()
print(end-start)
tracemalloc.reset_peak() 
tracemalloc.stop()
print(tree.export_text(skUnrestrictedModel))
tree.plot_tree(skUnrestrictedModel)
print(skUnrestrictedModel.score(X_test, y_test))


print("entropy: \n")
unrestrictedModel = DecisionTree(criterion='entropy')
unrestrictedModel.fit(X_train, y_train)
unrestrictedModel.print_tree()
print(unrestrictedModel.score(X_test, y_test))
print(unrestrictedModel.getTrainingTime())
print(unrestrictedModel.getMemoryUse())

skUnrestrictedModel = DecisionTreeClassifier(criterion='entropy')
start = time.time()
tracemalloc.start()
skUnrestrictedModel.fit(X_train, y_train)
print(tracemalloc.get_traced_memory()[1])
end = time.time()
print(end-start)
tracemalloc.reset_peak() 
tracemalloc.stop()
print(tree.export_text(skUnrestrictedModel))
tree.plot_tree(skUnrestrictedModel)
print(skUnrestrictedModel.score(X_test, y_test))


## Produce csv files for iris
iris_resultsDF, iris_library_resultsDF = splitter(X, y, [0.1, 0.2, 0.3, 0.4, 0.5])         
iris_resultsDF.to_csv("iris_results.csv")
iris_library_resultsDF.to_csv("iris_library_results.csv")



# Wine dataset

wine = np.array(pd.read_csv("wine.csv"))
y = wine[:,0]
X = wine[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("gini: \n")
unrestrictedModel = DecisionTree()
unrestrictedModel.fit(X_train, y_train)
unrestrictedModel.print_tree()
print(unrestrictedModel.score(X_test, y_test))
print(unrestrictedModel.getTrainingTime())
print(unrestrictedModel.getMemoryUse())

skUnrestrictedModel = DecisionTreeClassifier()
start = time.time()
tracemalloc.start()
skUnrestrictedModel.fit(X_train, y_train)
print(tracemalloc.get_traced_memory()[1])
end = time.time()
print(end-start)
tracemalloc.reset_peak() 
tracemalloc.stop()
print(tree.export_text(skUnrestrictedModel))
tree.plot_tree(skUnrestrictedModel)
print(skUnrestrictedModel.score(X_test, y_test))

#record as csv files
wine_resultsDF, wine_library_resultsDF = splitter(X, y, [0.1, 0.2, 0.3, 0.4, 0.5])    
wine_resultsDF.to_csv("wine_results.csv")
wine_library_resultsDF.to_csv("wine_library_results.csv")

         
#adult dataset, mix of continuous and categorical variables. sklearn cannot work on this!


adult = np.array(pd.read_csv('adult.csv'))[:1000,:]
y = adult[:,-1]
X = adult[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTree(criterion='entropy')
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(' ')
print(model.score(X_test,y_test))
print(model.getTrainingTime())
print(model.getMemoryUse())
model.print_tree()
print(model.getDepth())
