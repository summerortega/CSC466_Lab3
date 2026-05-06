import sys
import pandas as pd
from randomForest import RandomForest

import InduceC45
from InduceC45 import read_csv
def main(csv_file, output_file="evalReport.pdf", grid_file=None):
    X, y, a = read_csv(csv_file)