import pandas as pd
import numpy as np
import time

class ConditionContinuous:
    
    def __init__(self, col, val):
        
        self.col_name = col
        self.val = val
             
class Leaf:
    
    def __init__(self, label = None, class_probabilities = None):
        self.label = label
        self.class_probabilities = class_probabilities
         
class TreeNode:
    
    def __init__(self, condition):
        
        self.condition = condition
        self.children = []
        
def information_gain(y, y_children):
    
    measure_parent = gini_index(y)
    n = len(y)
    gain_split = measure_parent
    
    for y_i in y_children:
        
        n_i = len(y_i)
        gain_split -= (n_i/n) * gini_index(y_i)
    
    return gain_split

def gini_index(y):
    
    gini = 1
    classes = np.unique(y)
    
    for c in classes:
        gini-= (y == c).mean() ** 2
        
    return gini

def get_best_split(X, y, feature_subset=False):
    
    n_cols = X.shape[1]
    gain_max = -np.inf
    max_condition = None
    
    X_splits = []
    y_splits = []
    
    
    
    columns = np.array([*range(n_cols)])
    if feature_subset:
        idxs = get_features_subset(X, 8)
        columns = columns[idxs]
    
    
    for i in columns: 
       
        column = X[:,i]
        
        unique_vals = np.unique(column.round(2))
        
        for val in unique_vals:
            
            cond_true = column > val
            
            y_true = y[cond_true]
            y_false = y[~cond_true]
            y_children = (y_true, y_false)
            
            gain = information_gain(y, y_children)
            
            if gain > gain_max:
                max_condition = ConditionContinuous(i, val)
                gain_max = gain
                
                X_splits = []
                y_splits = []
                X_splits.append(X[cond_true])
                X_splits.append(X[~cond_true])
                y_splits.append(y_true)
                y_splits.append(y_false)
            
                
            
    
    return max_condition, X_splits, y_splits, gain_max
          
def build_tree(X, y, min_sample_split = 12, current_depth = 0, max_depth = 24, min_gain_threshold = 0.001,
              feature_subset = False):
    
    condition_1 = (len(np.unique(y)) == 1)
    condition_2 = (X.shape[0] < min_sample_split)
    condition_3 = current_depth == max_depth
    
    breaking_condition = condition_1 or condition_2 or condition_3
    
    
    if breaking_condition : ## all labels of the same class
        
        if condition_1:
            
            label = np.unique(y)[0]
            return Leaf(label)
        
        elif condition_2 or condition_3:
                
            classes, counts  = np.unique(y, return_counts = True)
            probabilities = counts/counts.sum()
            class_probabilities = dict(zip(classes, probabilities))
            return Leaf(class_probabilities=class_probabilities)
        
    
    current_depth +=1
    
    condition, X_splits, y_splits, max_gain = get_best_split(X, y, feature_subset)
    
    
    condition_4 = max_gain < min_gain_threshold
    
    if condition_4:
        
        classes, counts  = np.unique(y, return_counts = True)
        probabilities = counts/counts.sum()
        class_probabilities = dict(zip(classes, probabilities))
        return Leaf(class_probabilities=class_probabilities)

    node = TreeNode(condition)
  
    for X_i, y_i in zip(X_splits, y_splits):
        
        node.children.append(build_tree(X_i, y_i, min_sample_split, current_depth, 
                                        max_depth, feature_subset= True))
    
    return node

def predict_row(row, root):
    
    if isinstance(root, Leaf):
        
        if root.label:
            return root.label
        
        elif root.class_probabilities:
            classes = list(root.class_probabilities.keys())
            probs = list(root.class_probabilities.values())
            return np.random.choice(classes, 1, p = probs)[0]
        
    col = root.condition.col_name
    val = root.condition.val
    left = root.children[0]
    right = root.children[1]
    
    if row[col] > val:
        return predict_row(row, left)
    else:
        return predict_row(row, right)
    
def predict_tree(X_test, tree):
    
    test_pred = []
    for row in X_test:
        test_pred.append(predict_row(row, tree))
        
    
    return np.array(test_pred)

def get_bootstrapped_data(X, y):
    
    n_rows = X.shape[0]
    sample_idxs = np.random.choice(np.arange(n_rows), n_rows) ## samples indices with replacement
    
    return X[sample_idxs,:], y[sample_idxs]

def get_features_subset(X, n_features):
    
    total_n_features = X.shape[1]
    features_idxs = np.random.choice(np.arange(total_n_features), n_features, replace = False)
    
    features_idxs.sort()
    
    return features_idxs

def build_forest(X, y, n_estimators, min_sample_split = 12,
                       max_depth = 10, min_gain_threshold = 0.001, 
                       feature_subset = False):
    
    trees = []
    for i in  range(n_estimators):
        
        start = time.time()
        
        bd, y_train = get_bootstrapped_data(X, y)
        
         
              
        dt = build_tree(bd, y_train, max_depth= max_depth, 
                                        min_sample_split= min_sample_split,
                                        min_gain_threshold= min_gain_threshold, 
                                        feature_subset = feature_subset)
        trees.append(dt)
        print(i+1," tree build in {} s".format(round(time.time()-start, 2)))
    
    return trees
    
def most_frequent(preds):

    ### take majority vote of trees

    labels, counts = np.unique(preds, return_counts = True)
    return labels[np.argmax(counts)]

def predict(X_test, trees):
    preds = np.zeros((X_test.shape[0], len(trees)))
    
    for i,tree in enumerate(trees):
        preds[:,i] = predict_tree(X_test, tree)
    
    
    rf_preds = np.apply_along_axis(most_frequent, 1, preds.astype(int))

    return rf_preds