import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set.


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    score : float
        Average CV score
    timeElapsed: float
        Time it took to run this function
    """

    timeElapsed = time.monotonic()
    kFold = KFold(n_splits=k)
    scores = cross_val_score(model, xFeat, y, scoring='neg_mean_squared_error', cv=kFold)
    print(scores)
    score = sum(scores) / (len(scores))
    timeElapsed = time.monotonic() - timeElapsed
    return score, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """

    Parameters
    ----------
    model : RandomForestClassifier object
        An instance of the random forest classifier
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainErr : float
        The AUC of the model evaluated on the training data.
    testErr : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict(xTrain)
    yHatTest = model.predict(xTest)
    # calculate auc for training
    print(yTrain)
    print(yHatTrain)
    trainErr = mean_squared_error(yTrain, yHatTrain)
    # calculate auc for test dataset
    testErr = mean_squared_error(yTest, yHatTest)
    return trainErr, testErr


def transform(y):
    for i in range(len(y)):
        if y[i] < 0 or y[i] < 0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y


def countEst(y, yTrue):
    count = 0
    for i in range(len(y)):
        if y[i] == yTrue[i]:
            count += 1
    return count

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainFile",
                        default="xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="xTest.csv",
                        help="filename of the test data")
    parser.add_argument("--trainy",
                        default="yTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testy",
                        default="yTest.csv",
                        help="filename of the training data")
    args = parser.parse_args()
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    yTrain = pd.read_csv(args.trainy)
    yTest = pd.read_csv(args.testy)
    yTrain, yTest = np.ravel(yTrain), np.ravel(yTest)

    trainSc = list()
    # SGD Classifier
    model0 = SGDClassifier(penalty='l1', learning_rate='optimal', random_state=36)
    model0.fit(xTrain, yTrain)
    trainSc.append(model0.score(xTrain, yTrain))
    trainSc.append(model0.score(xTest, yTest))

    model0 = SGDClassifier(penalty='l2', learning_rate='optimal', random_state=36)
    model0.fit(xTrain, yTrain)
    trainSc.append(model0.score(xTrain, yTrain))
    trainSc.append(model0.score(xTest, yTest))

    timeElapsed = time.monotonic()
    model0 = SGDClassifier(penalty='elasticnet', learning_rate='optimal', random_state=36)
    model0.fit(xTrain, yTrain)
    print('time of SGD: ', time.monotonic() - timeElapsed)
    trainSc.append(model0.score(xTrain, yTrain))
    trainSc.append(model0.score(xTest, yTest))

    plot0 = sns.barplot(x=['l1', 'l1', 'l2', 'l2', 'elasticnet', 'elasticnet'], y=trainSc, hue=['train', 'test', 'train', 'test', 'train', 'test'])
    plot0.set_yscale('log')
    plot0.set_xlabel('regularization term')
    plot0.set_ylabel('accuracy')
    plt.show()

    trainSc0 = model0.score(xTrain, yTrain)
    testSc0 = model0.score(xTest, yTest)
    print('SGD Classifier Train Acc:', trainSc0)
    print('SGD Classifier Test Acc:', testSc0)

    # Regular Linear Regression
    timeElapsed = time.monotonic()
    model1 = LinearRegression()
    model1.fit(xTrain, yTrain)
    print('time of LinearRegression: ', time.monotonic() - timeElapsed)
    yHatTrain1 = model1.predict(xTrain)
    yHatTest1 = model1.predict(xTest)
    trainSc1 = countEst(transform(yHatTrain1), yTrain) / len(yTrain)
    testSc1 = countEst(transform(yHatTest1), yTest) / len(yTest)
    print('Regular Linear Rregression Train Acc:', trainSc1)
    print('Regular Linear Rregression Test Acc:', testSc1)

    # Find the best alpha value for Lasso Regression
    times = list()
    for i in range(10):
        alpha = 0.1
        timeElapsed = time.monotonic()
        model = Lasso(alpha=alpha, random_state=334)
        model.fit(xTrain, yTrain)
        yHatTrain = model.predict(xTrain)
        times.append(time.monotonic() - timeElapsed)
        alpha += 0.1
    plot = sns.lineplot(x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], y=times)
    plot.set_xlabel("alpha")
    plot.set_ylabel("time")
    plt.show()

    # Lasso Regression
    timeElapsed = time.monotonic()
    model2 = Lasso(alpha=1.0, random_state=34)
    model2.fit(xTrain, yTrain)
    print('time of Lasso: ', time.monotonic() - timeElapsed)
    yHatTrain2 = model2.predict(xTrain)
    yHatTest2 = model2.predict(xTest)
    trainSc2 = countEst(transform(yHatTrain2), yTrain) / len(yTrain)
    testSc2 = countEst(transform(yHatTest2), yTest) / len(yTest)
    print('Lasso coefficients: ', model2.coef_)
    print('Lasso Rregression Train Acc:', trainSc2)
    print('Lasso Rregression Test Acc:', testSc2)

    timeElapsed = time.monotonic()
    model3 = RandomForestClassifier(n_estimators=100, random_state=34)
    model3.fit(xTrain, yTrain)
    print('time of RandomForest: ', time.monotonic() - timeElapsed)
    trainSc3 = model3.score(xTrain, yTrain)
    testSc3 = model3.score(xTest, yTest)
    print('RandomForest Train Acc:', trainSc3)
    print('RandomForest Test Acc:', testSc3)
    trainSc = list()
    for i in range(2):
        c = 'gini'
        model = RandomForestClassifier(n_estimators=100, criterion=c, random_state=34)
        model.fit(xTrain, yTrain)
        trainSc.append(model.score(xTest, yTest))
        c = 'entropy'
    plot1 = sns.barplot(x=['gini', 'entropy'], y=trainSc).set_yscale('log')
    plt.show()
    # use 5-fold validation
    score0, time0 = kfold_cv(model0, xTrain, yTrain, 5)
    score1, time1 = kfold_cv(model1, xTrain, yTrain, 5)
    score2, time2 = kfold_cv(model2, xTrain, yTrain, 5)
    score3, time3 = kfold_cv(model3, xTrain, yTrain, 5)
    trainErr, testErr = sktree_train_test(model0, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['SGD', 1 + score0, time0],
                           ['LinearRegression', 1 + score1, time1],
                           ['Lasso', 1 + score2, time2],
                           ['RandomForest', 1 + score3, time3]],
                          columns=['Strategy', 'score', 'Time'])
    print(perfDF)
    print(trainErr)
    print(testErr)


if __name__ == "__main__":
    main()
