import argparse
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_barplot(feature_name, spotify_data):
    g = sns.factorplot(x="instrumentalness", y="popularity", data=spotify_data, kind="bar", size=6)
    g.set_ylabels("Instrumentalness vs Popularity")
    plt.show()


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data
    testDF : pandas dataframe
        Test data
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    trainDF = trainDF.to_numpy()
    testDF = testDF.to_numpy()
    mmScale = StandardScaler()
    mmScale.fit(trainDF)
    trainDF = mmScale.transform(trainDF)
    testDF = mmScale.transform(testDF)
    trainDF = pd.DataFrame(data=trainDF[0:, 0:])
    testDF = pd.DataFrame(data=testDF[0:, 0:])
    return trainDF, testDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spdata",
                        default="data.csv",
                        help="filename for features of the training data")
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("outyTrain",
                        help="filename of the updated training data")
    parser.add_argument("outyTest",
                        help="filename of the updated test data")

    args = parser.parse_args()
    # load the train and test data
    spdata = pd.read_csv(args.spdata)
    spdata = spdata.drop(['id', 'release_date', 'artists', 'name'], axis=1)
    print(spdata.shape[0], spdata.shape[1])
    spdata = spdata[spdata['year'] >= 1980]
    print(spdata.shape[0], spdata.shape[1])
    y = spdata['popularity']
    xFeat = spdata.drop(['popularity'], axis=1)

    pearson_matrix = spdata.corr(method='pearson')
    plt.title('Correlation Map')
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(pearson_matrix,
                     xticklabels=pearson_matrix.columns,
                     yticklabels=pearson_matrix.columns,
                     annot=True)
    plt.show()
    targetVar = abs(pearson_matrix["popularity"])

    relevantFeat = targetVar[targetVar > 0.1]
    print(relevantFeat)
    print(spdata[["explicit", "year", "loudness", "danceability", "instrumentalness"]].corr())

    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=0.33)

    xNewTrain = xTrain.drop(['acousticness', 'loudness', 'duration_ms', 'energy', 'key', 'liveness', 'mode', 'speechiness', 'tempo', 'valence'], axis=1)
    xNewTest = xTest.drop(['acousticness', 'loudness', 'duration_ms', 'energy', 'key', 'liveness', 'mode', 'speechiness', 'tempo', 'valence'], axis=1)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)
    yTrain.to_csv(args.outyTrain, index=False)
    yTest.to_csv(args.outyTest, index=False)


if __name__ == "__main__":
    main()
