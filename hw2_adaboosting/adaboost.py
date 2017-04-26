import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def adaboost(x, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    epsilon = 1.0e-6
    w = np.array([1.0/len(y) for i in range(len(y))])
    for itr in range(num_iter):
        algo = DecisionTreeClassifier(max_depth=1)
        algo.fit(x, y, sample_weight=w)
        trees.append(algo)
        y_pred = algo.predict(x)

        find_unmatched = np.array(y) != np.array(y_pred)
        error = sum(find_unmatched * w)/sum(w)
        if abs(error) < epsilon: # if all observations classified correctly
            trees_weights.append(100) # approaches infinite
            return trees, trees_weights
        alpha_m = np.log((1 - error) / error)
        trees_weights.append(alpha_m)
        w *= np.exp(find_unmatched * alpha_m)
    return trees, trees_weights

def adaboost_predict(x, trees, trees_weights):
    """Given X, trees and weights predict Y

    assume Y in {-1, 1}^n
    """
    Yhat = []
    for itr in range(len(trees)):
        prediction = trees[itr].predict(x)
        Yhat.append(np.array(prediction * trees_weights[itr]))
    Yhat = np.sign( sum(Yhat) )
    return Yhat

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    df = pd.read_table(filename, sep=",", header=None)
    x = df.iloc[:,0:-1]
    y = df.iloc[:, -1]
    return x, y

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]

def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y)) 

def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train spambase.train --test spambase.test --numTrees 4
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])

    x_train, y_train = parse_spambase_data(train_file)
    x_test, y_test = parse_spambase_data(test_file)

    trees, trees_weights = adaboost(x_train, new_label(y_train), num_trees)

    y_pred_train = adaboost_predict(x_train, trees, trees_weights)
    y_pred_test = adaboost_predict(x_test, trees, trees_weights)

    output = pd.DataFrame(old_label(y_pred_train))
    output = pd.concat([x_train, y_train, output.astype(int)], axis = 1)
    output.to_csv("prediction.txt", sep = ",", header = False, index = False)

    acc_test = accuracy(y_test, old_label(y_pred_test))
    acc = accuracy(y_train, old_label(y_pred_train))
    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

if __name__ == '__main__':
    main()
    # python adaboost.py --train spambase.train  --test spambase.test --numTrees 4
