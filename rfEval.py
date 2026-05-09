import sys
import pandas as pd
from randomForest import RandomForest
from readcsv import read_csv

def main(csv_file, output_file="evalReport.pdf", grid_file=None):
    x, y, a = read_csv(csv_file)
    