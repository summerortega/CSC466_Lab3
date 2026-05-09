import sys
import pandas as pd
from randomForest import RandomForest
from readcsv import read_csv

def main(csv_file, output_file="evalReport.pdf", grid_file=None):
    #read the data
    x, y, a = read_csv(csv_file)
    #perform the 80-20 split on the data
    x_test = x.sample(frac=0.2)
    y_test = y.loc[x_test.index]
    x_train = x.drop(x_test.index)
    y_train = y.loc[x_train.index]
    #perform the hyperparameter tuning