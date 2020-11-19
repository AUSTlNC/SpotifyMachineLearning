import argparse
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

    model1 = LinearRegression()

    model1.fit(xTrain, yTrain)

    yHatTrain1 = model1.predict(xTrain)
    yHatTest1 = model1.predict(xTest)

    trainEr1 = mean_squared_error(yTrain, yHatTrain1)
    testEr1 = mean_squared_error(yTest, yHatTest1)

    print(trainEr1)
    print(testEr1)


if __name__ == "__main__":
    main()
