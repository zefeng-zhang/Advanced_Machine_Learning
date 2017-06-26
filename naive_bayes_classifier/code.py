from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from collections import defaultdict
import numpy as np
import pandas as pd

def parse_data(filename):
    """
    return training and testing data sets
    """
    df = pd.read_csv(filename, header=None)
    train, test = train_test_split(df, test_size=1.0/4.0)
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    return train, test, x_train, y_train, x_test, y_test

def get_summaries(df):
    """
    For each class/category, return mean and std of all features
    """
    categories = df.iloc[:, -1].unique()
    summaries = defaultdict(list)
    for cat in categories:
        sub = df.ix[df.iloc[:, -1] == cat, :]
        print "\ncategory, proportion"
        print cat, 1.0 * len(sub.index) / len(df)
        for i in range(len(sub.columns) - 1):
            summaries[cat].append([np.mean(sub.iloc[:, i]), np.std(sub.iloc[:, i])])
            print "{:.1f}, {:.1f}, ".format(np.mean(sub.iloc[:, i]), np.std(sub.iloc[:, i])),
    return summaries

def gaussian_probability(x, mean, std):
    """
    Given mean and std, return the probability of getting x
    """
    exponent = np.exp(-(x - mean)**2.0 / 2 / std**2.0)
    return 1.0/ np.sqrt(2.0*np.pi) / std * exponent

def row_label(summaries, row):
    """
    Given a new observation, return the predicted label
    """
    probability = {}
    for class_value, class_summary in summaries.iteritems():
        probability[class_value] = 1
        for i in range(len(class_summary)):
            mean, std = class_summary[i]
            probability[class_value] *= gaussian_probability(row[i], mean, std)
    return max(probability, key=probability.get)

def prediction(summaries, test_x):
    """
    Given a testing data set, return a list of predicted labels
    """
    test_x['y'] = test_x.apply(lambda x: row_label(summaries, x), axis=1)
    return test_x['y']

def report(y, y_pred):
    """
    Return prediction accuracy
    """
    print "\naccuracy = {}".format( 1.0 * np.sum(np.array(y) == np.array(y_pred)) / len(y) )

if _name_ == "__main__":
    train, test, x_train, y_train, x_test, y_test = parse_data("pima-indians-diabetes.data.txt")
    summaries = get_summaries(train)
    y_pred = prediction(summaries, x_test)
    report(y_test, y_pred)
