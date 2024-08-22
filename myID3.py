#!/usr/bin/env python
# coding: utf-8

##imports
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm

##imports for sk comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
##from sklearn.metrics import classification_report

# ## Fetch data 
print("Reading Data")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=int(sys.argv[1])) #####hyperparameter

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

print("Finished Reading Data")


## keep only the first n that are inserted as hyperparameter
x_train = x_train[:int(sys.argv[2])]
y_train = y_train[:int(sys.argv[2])]


# ## Create Vocabulary

print("Creating Vocabulary")

vocabulary = list()
for text in x_train:
  tokens = text.split()
  vocabulary.extend(tokens)

##vocabulary = set(vocabulary[int(sys.argv[3]): 1 + int(sys.argv[3]) + int(sys.argv[1])])
vocabulary = set(vocabulary)
##voc = set(list(vocabulary)[int(sys.argv[3]): int(sys.argv[3]) + int(sys.argv[1])]) ###########An thelw na sfaksw tiw pio syxna emfanizomenes
vocabulary = set(list(vocabulary)[int(sys.argv[3]):])
print(len(vocabulary))
print("Vocabulary created")


# ## Create binary vectors 
print("Creating binary vectors ")
x_train_binary = list()
x_test_binary = list()

for text in tqdm(x_train):
  tokens = text.split()
  binary_vector = list()
  for vocab_token in vocabulary:
    if vocab_token in tokens:
      binary_vector.append(1)
    else:
      binary_vector.append(0)
  x_train_binary.append(binary_vector)

x_train_binary = np.array(x_train_binary)

for text in tqdm(x_test):
  tokens = text.split()
  binary_vector = list()
  for vocab_token in vocabulary:
    if vocab_token in tokens:
      binary_vector.append(1)
    else:
      binary_vector.append(0)
  x_test_binary.append(binary_vector)

x_test_binary = np.array(x_test_binary)

print("Binary vectors created")


# ## Entropy Implementation 

def entropy(class_y):
    
    if len(class_y) <=1: ## Handling if there is only 1 or 0 labels
        return 0
    
    total_count = np.bincount(class_y) # count occurence of each element
    probabilities = total_count[np.nonzero(total_count)] / len(class_y) # Find the probabilities
    if len(probabilities) <= 1 : ## Handling if the length of the probabilities is less than or equal to 1
        return 0
    return - np.sum(probabilities * np.log2(probabilities)) ## Entropy equation for 2 categories


# ## Information Gain Implementation

def information_gain(previous_y, current_y):
    ## IG = H(Y) - CE(Y|X)
    conditional_entropy = 0 
    for y in current_y:
        conditional_entropy += (entropy(y)*len(y)/len(previous_y))
    info_gain = entropy(previous_y) - conditional_entropy
    return info_gain


# ## The Decision Tree ID3 algorithm

# ### Partition classes (for numerical attribute)

def partition_classes(X, y, split_attribute):
   
   X = np.array(X)
   column_split = X[:,split_attribute]
   X_left=[]
   y_right = []
   X_right = []
   y_left = []
   
   counter=0 ## Counter for appending

   for i in column_split:
       if i == 1:
           X_left.append(X[counter])
           y_left.append(y[counter])
       else:
           X_right.append(X[counter])
           y_right.append(y[counter])
       counter+=1

   return X_left, X_right, y_left, y_right

## Best feature

def find_best_feature(X, y):
    
    best_info_gain = 0
    best_feature = 0
    for feature_index in range(len(X[0])): ## Loop over the features , find the one with the best information gain
        current_X_left, current_X_right, current_y_left, current_y_right = partition_classes(X, y, feature_index) ## Call the partition classes function
        current_y = []
        current_y.append(current_y_left)
        current_y.append(current_y_right)
        current_info = information_gain(y,current_y) ## Calculate the information gain
        if current_info > best_info_gain:
            best_info_gain = current_info
            best_feature = feature_index
    return best_feature


# ## Main Algorithm 

class MyDecisionTree(object):
    def __init__(self, max_depth= None):
        self.tree = {} ##create empty tree
        self.residual_tree = {} ## For prediction
        self.max_depth = max_depth

        
    def fit(self, X, y, depth):

        # base cases
        unique_labels = np.unique(y) 
        unique_labels, counts_unique_labels = np.unique(y, return_counts=True)
        index = counts_unique_labels.argmax()## returns the index of the maximum
        #prepruning
        if ( counts_unique_labels[index]/float(sum(counts_unique_labels)) >=0.95 ) or (depth == self.max_depth)  : 
            classification = unique_labels[index]    
            return classification ##epistrefei to symperasma an tha einai 0 H 1
    
        best_feat  = find_best_feature(X, y) ## Find best feature 
        X_left, X_right, y_left, y_right = partition_classes(X, y, best_feat) ## Partition on the best feature 
        
        question = "{} {} == {}".format(list(vocabulary)[best_feat], best_feat, 1) ## Represnt the sub-tree as a question and an answer
        node = {question: []} ## node is a dictionary. As key is set the question
        
        # find answers (recursion)
        depth+=1 
        yes_answer = self.fit(X_left,y_left, depth)  ## left sub tree
        no_answer = self.fit(X_right, y_right, depth) ## right sub tree

        if yes_answer == no_answer: ## Both trees are the same
            node = yes_answer
        else:
            #xtizw to dentro
            node[question].append(yes_answer) ## Append the questions to the answers
            node[question].append(no_answer)
        self.tree = node ## The tree is equal to the node
        return node
        
    def predict(self, record,flag=1):
        """  
        record: x_test_binary[i]
        Output:True if the predicted class label is 1, else false
        """
        if flag == 1: ## First time
            self.residual_tree = self.tree

        question = list(self.residual_tree.keys())[0] #pairnei to label
        feature, index_feature, comparison, value = question.split() ## Split the question to get the feature and its value

        
        if record[int(index_feature)] == int(value) : 
            answer = self.residual_tree[question][0]  ## Left sub tree
        else:
            answer = self.residual_tree[question][1] ## Right sub-tree
                
        # base case
        if not isinstance(answer, dict):  ## If we have the answer
            return answer
        # recursion
        else:
            self.residual_tree = answer ## The residual tree is the answer !
            return self.predict(record,0) ## have flag = 0 so the residual tree is our sub-problem


# ## Evaluation implementation

def DecisionTreeEvalution(id3,X,y, verbose=False):

    y_predicted = []
    for record in X: 
        y_predicted.append(id3.predict(record))
    
    # Comparing predicted and true labels
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for prediction, truth in zip(y_predicted, y):
        if prediction == truth :
            if prediction == 1:
                tp +=1 
            else:
                tn +=1
        else:
            if prediction == 1:
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
    
    if verbose:
        print("accuracy: %.4f" % accuracy)
        print("precision: %.4f" % precision)
        print("recall: %.4f" % recall)
        print("f1: %.4f" % f1)
    return accuracy, precision, recall, f1


if(sys.argv[4] == "1"):
    max_depth = int(sys.argv[5])
    inital_depth = 0
    id3_dt = MyDecisionTree(max_depth = max_depth)

    ## Building the tree
    print("Fitting the decision tree")
    id3_dt.fit(x_train_binary, y_train, inital_depth)

    ##print(id3_dt.tree)

    ## Evaluating the tree
    print("MyID3 Evaluation with test data")
    DecisionTreeEvalution(id3_dt,x_test_binary,y_test, True)

    print("MyID3 Evaluation with train data")
    DecisionTreeEvalution(id3_dt,x_train_binary,y_train, True)

    print("////////////////////////////////////////////////////////")
    print("Comparison with Scikit-learn")

    
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(x_train_binary, y_train)

    print("Scikit-learn Evaluation with test data")
    y_pred = dt.predict(x_test_binary)

    print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
    print("Precision: %.4f" % precision_score(y_test, y_pred))
    print("Recall: %.4f" % recall_score(y_test, y_pred))
    print("F1: %.4f" % f1_score(y_test, y_pred))
    ##print(classification_report(y_test[:10000], y_pred))

    print("Scikit-learn Evaluation with train data")
    y_pred = dt.predict(x_train_binary)

    print("Accuracy: %.4f" % accuracy_score(y_train, y_pred))
    print("Precision: %.4f" % precision_score(y_train, y_pred))
    print("Recall: %.4f" % recall_score(y_train, y_pred))
    print("F1: %.4f" % f1_score(y_train, y_pred))