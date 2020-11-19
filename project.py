import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


def normalize_features(xTrain, xTest):
    """
    Normalize the features in the wine quality dataset.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data

    Returns
    -------
    xTrain : nd-array with shape n x d
        Normalized train data.
    xTest : nd-array with shape m x d
        Normalized test data.
    """
    stdScale = StandardScaler()
    stdScale.fit(xTrain)
    xTrain = stdScale.transform(xTrain)
    xTest = stdScale.transform(xTest)
    return xTrain, xTest


def logistic_prob(xTrain, yTrain, xTest):
    """
    Train the unregularized logistic regression model on the normalized dataset.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d-array with shape n x 1
        Training data labels
    xTest  : md-array with shape m x d
        Testing data

    Returns
    -------
    prob : The probabilities on the test dataset using the unregularized logistic regression model trained with the normalized dataset.
    """
    model = LogisticRegression(penalty='none').fit(xTrain, yTrain)
    prob = model.predict_proba(xTest)
    return prob


def findBestK(xTrain):
    varExplained = 0.0
    i = 1
    while varExplained < 0.95:
        train_pca = PCA(n_components=i)
        xTrain_pca = train_pca.fit_transform(xTrain)
        varExplained = sum(train_pca.explained_variance_ratio_)
        i += 1
    return i - 1


def pca(xTrain, xTest):

    xTrain, xTest = normalize_features(xTrain, xTest)
    k = findBestK(xTrain)
    train_pca = PCA(n_components=k)
    xTrain_pca = train_pca.fit_transform(xTrain)
    xTest_pca = train_pca.transform(xTest)
    com = train_pca.components_
    return xTrain_pca, xTest_pca, com


def logisticRoc(xTrain, yTrain, xTest, yTest):
    # Normalized the dataset
    xTrain1, xTest1 = normalize_features(xTrain, xTest)
    # PCA dataset
    xTrain2, xTest2, components = pca(xTrain, xTest)
    # Probabilities
    prob1 = logistic_prob(xTrain1, yTrain, xTest1)
    prob2 = logistic_prob(xTrain2, yTrain, xTest2)

    fpr1, tpr1, thresholds1 = metrics.roc_curve(yTest, prob1[:, 1])
    fpr2, tpr2, thresholds2 = metrics.roc_curve(yTest, prob2[:, 1])
    fpr = np.concatenate((fpr1, fpr2), axis=None)
    tpr = np.concatenate((tpr1, tpr2), axis=None)
    names = list()
    for i in range(len(fpr1)):
        names.append('Normalized')
    for i in range(len(fpr2)):
        names.append('Normalized PCA')
    plot = sns.lineplot(x=fpr, y=tpr, hue=names)
    plot.set_title("ROC Curve for Normalized dataset and Normalized PCA dataset")
    plot.set_xlabel("False Positive Rate")
    plot.set_ylabel("True Positive Rate")
    plt.show()
    print(components)


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q1xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q1yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q1xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q1yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    yTrain, yTest = np.ravel(yTrain), np.ravel(yTest)
    logisticRoc(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()