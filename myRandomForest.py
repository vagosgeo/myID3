##imports
import sys
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm_notebook as tqdm

##imports for sk comparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

##imports from myID3
from myID3 import MyDecisionTree ##My ID3
from myID3 import x_test_binary
from myID3 import x_train_binary
from myID3 import y_test
from myID3 import y_train


class MyRandomForest(object):
    def __init__(self, n_estimators=50, max_depth=None, max_features=0.7):
        self.n_estimators = n_estimators ## num of trees
        self.max_depth = max_depth 
        self.max_features = max_features ## the percentage of the random features we will use
        self.bootstraps_row_indices = [] ## Which rows are in the current bootstrap
        self.feature_indices = [] 
        self.decision_trees = [] ## random forest
        
        ## create my trees
        for i in range(n_estimators):
            self.decision_trees.append(MyDecisionTree(max_depth=max_depth)) 
        
    def _bootstrapping(self, num_training, num_features): 
        sample_size = list(range(num_training)) ##{0,1,2,..., num_training -1}
        row_idx = np.random.choice(sample_size,num_training) ## Create random row indices
        col_idx = np.random.permutation(num_features)[:int(num_features*self.max_features)] ## shuffle and select the first n
        return row_idx, col_idx
            
    def bootstrapping(self, num_training, num_features):
        """
        Initializing the bootstap datasets for each tree
        """
        
        for i in range(self.n_estimators):
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            
            
    def fit(self, X, y):
        """
        Train decision trees using the bootstrapped datasets.
        """
        
        num_training, num_features = X.shape
        self.bootstrapping(num_training,num_features) ## initialize the bootstrapping
        for i in range((self.n_estimators)): ## loop over the trees 
            current_bootstraps_row_indices = self.bootstraps_row_indices[i]
            current_feature_indices = self.feature_indices[i]
            current_X = X[current_bootstraps_row_indices[:,np.newaxis], current_feature_indices] ## data for this tree
            current_y = y[current_bootstraps_row_indices]
            current_dt = self.decision_trees[i]
            current_dt.fit(current_X,current_y, 0) ## 0 for the initial depth
            print("Current Tree to fit : ", i+1) ## Which tree we are using


    def RFpredict(self, X, y):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for i in range(len(X)): ## Loop over the full dataset
            predictions = []
            for t in range(self.n_estimators): ## loop over each decision tree
                predictions.append(self.decision_trees[t].predict(X[i][self.feature_indices[t]])) ## Predict
            unique_labels, counts_unique_labels = np.unique(predictions, return_counts=True)
            index = counts_unique_labels.argmax()## returns the index of the maximum
            if unique_labels[index] == y[i] :
                if unique_labels[index] == 1:
                    tp +=1 
                else:
                    tn +=1
            else:
                if unique_labels[index] == 1:
                    fp +=1 
                else:
                    fn +=1 
    
        # Accuracy
        accuracy = (tp + tn )/ float(tp + tn + fn + fp ) 

        # Precision
        precision = tp / float(tp + fp )

        # Recall
        recall = tp / float(tp + fn)

        # F1
        f1 = (2*precision*recall)/(precision + recall)

        print("accuracy: %.4f" % accuracy)
        print("precision: %.4f" % precision)
        print("recall: %.4f" % recall)
        print("f1: %.4f" % f1)
        
        return accuracy, precision, recall, f1

n_estimators = int(sys.argv[6])
max_depth = int(sys.argv[5])
max_features = 0.8
random_forest = MyRandomForest(n_estimators, max_depth, max_features)
random_forest.fit(x_train_binary, y_train)
print("MyRandomForest Evaluation with test data")
random_forest.RFpredict(x_test_binary, y_test)
print("MyRandomForest Evaluation with train data")
random_forest.RFpredict(x_train_binary, y_train)

skRandomForest = RandomForestClassifier(max_depth=max_depth, n_estimators =n_estimators , criterion = 'entropy')
skRandomForest.fit(x_train_binary, y_train)
print("Scikit-learn Evaluation with test data")
y_pred = skRandomForest.predict(x_test_binary)
print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("Precision: %.4f" % precision_score(y_test, y_pred))
print("Recall: %.4f" % recall_score(y_test, y_pred))
print("F1: %.4f" % f1_score(y_test, y_pred))
print("Scikit-learn Evaluation with train data")
y_pred = skRandomForest.predict(x_train_binary)
print("Accuracy: %.4f" % accuracy_score(y_train, y_pred))
print("Precision: %.4f" % precision_score(y_train, y_pred))
print("Recall: %.4f" % recall_score(y_train, y_pred))
print("F1: %.4f" % f1_score(y_train, y_pred))