import argparse

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data_by_year",
                        help="filename for features of the training data")
    args = parser.parse_args()
    dby = file_to_numpy(args.data_by_year)
    year = list()
    pplrt = list()
    acou = list()
    dance = list()
    duration = list()
    energy = list()
    instru = list()
    live = list()
    loud = list()
    speech = list()
    tempo = list()
    valence = list()
    key = list()
    names = list()
    att = ['popularity', 'acousticness', 'danceability',
           'duration', 'energy', 'instrumentalness',
           'liveliness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'key']
    for i in range(len(dby)):
        pplrt.append(dby[i][-3]/10)
        acou.append(dby[i][1])
        dance.append(dby[i][2])
        duration.append(dby[i][3]/100000)
        energy.append(dby[i][4])
        instru.append(dby[i][5])
        live.append(dby[i][6])
        loud.append(dby[i][-7]/10)
        speech.append(dby[i][-6])
        tempo.append(dby[i][-5]/100)
        valence.append(dby[i][-4])
        key.append(dby[i][-2])
    for i in range(12):
        for j in range(len(dby)):
            year.append(int(dby[j][0]))
            names.append(att[i])
    print(len(names))
    print(len(year))
    plotList = pplrt + acou + dance + duration + energy + instru + live + loud + speech + tempo + valence + key
    plot = sns.lineplot(x=year, y=plotList, hue=names)
    plot.set_title("Averaged features of years")
    plot.set_xlabel("Years")
    plot.set_ylabel("Average popularity")
    plt.show()

if __name__ == "__main__":
    main()