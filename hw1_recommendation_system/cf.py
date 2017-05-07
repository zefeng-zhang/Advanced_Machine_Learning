import argparse
import re
import os
import csv
import math
import collections as coll
import pandas as pd
import numpy as np

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries

    Input: filename

    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    df = pd.read_table(filename, sep=",", header=None)
    df.columns = ['MovieId', 'CustomerId', 'Rating']
    unique_CustomerId = df['CustomerId'].unique()
    unique_MovieId = df['MovieId'].unique()

    for customerId in unique_CustomerId:
        df_customerId = df.loc[df['CustomerId'] == int(customerId), :]
        MovieId = df_customerId['MovieId']
        Rating = df_customerId['Rating']
        user_ratings[int(customerId)] = dict(zip(MovieId, Rating))

    for MovieId in unique_MovieId:
        df_MovieId = df.loc[df['MovieId'] == int(MovieId), :]
        CustomerId = df_MovieId['CustomerId']
        Rating = df_MovieId['Rating']
        movie_ratings[int(MovieId)] = dict(zip(CustomerId, Rating))

    return user_ratings, movie_ratings

def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings

    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = {}
    for user, movie_ratings in user_ratings.items():
        ave_ratings[user] = np.mean(movie_ratings.values())
    return ave_ratings

def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users
        Complexity: N^2 * K

        Input: d1, d2, (dictionary of user ratings per user) 
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    epsilon = 1.0e-6
    numerator = 0.0
    var1 = 0.0
    var2 = 0.0
    for item in d1:
        if item in d2:
        # if two users share the same items
            numerator += (d1[item] - ave_rat1) * (d2[item] - ave_rat2)
            var1 += (d1[item] - ave_rat1) ** 2.0
            var2 += (d2[item] - ave_rat2) ** 2.0
    denominator = (var1 * var2) ** 0.5
    if abs(denominator) > epsilon:
        return numerator / denominator
    else:
        return 0.0

def main():
    """
    This function is called from the command line via
    
    python cf.py --train TrainingRatings.txt --test TestingRatingsMedium.txt
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]

    epsilon = 1.0e-6
    RMSE = 0.0
    MAE = 0.0
    testing = pd.read_table(test_file, sep=",", header=None)
    # Two dictionaries "user_ratings" and "movie_ratings"
    user_ratings, movie_ratings = parse_file(train_file)
    # Compute dictionary "ave_ratings"
    ave_ratings = compute_average_user_ratings(user_ratings)

    f = open("prediction.txt", 'w')
    for index in testing.index: # complexity: K * N
        testing_data =testing.iloc[index, :]
        k = testing_data[0] # MovieId
        i = testing_data[1] # CustomerId
        Rik = testing_data[2] # Rating

        """ Predict movie ratings based on user similarity """
        w_sum = 0.0
        term2 = 0.0
        # complexity: N
        for j in movie_ratings[k]: 
            if j != i:
                wij = compute_user_similarity(user_ratings[i], user_ratings[j], \
                                              ave_ratings[i], ave_ratings[j]) 
                term2 += wij * (user_ratings[j][k] - ave_ratings[j])
                w_sum += abs(wij)
        if abs(w_sum) > epsilon:
            Rik_fitted = ave_ratings[i] + 1.0 / w_sum * term2
        else: 
        # Cold start: new user who only watches one movie
            Rik_fitted = ave_ratings[i]

        f.write("{}, {}, {}, {}\n".format(int(k), int(i), Rik, Rik_fitted))
        RMSE += (Rik - Rik_fitted) ** 2.0
        MAE += abs(Rik - Rik_fitted)

    print "RMSE %.4f" % (RMSE / float(len(testing.index))) ** 0.5
    print "MAE %.4f" % ( MAE / float(len(testing.index)) )
    f.close()

if __name__ == '__main__':
    main()
    # commands:
    # python cf.py --train TrainingRatings.txt --test TestingRatingsMedium.txt
    # python cf.py --train tinyTraining.txt --test tinyTesting.txt

