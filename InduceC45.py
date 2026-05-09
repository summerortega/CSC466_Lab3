import pandas as pd
import json
import sys
from c45 import C45Tree
from readcsv import read_csv

def main(csv_path:str, save_file_path:str = None) -> None:
    #get training set x, class var y, and attribute series a
    x, y, a = read_csv(csv_path)
    #instantiate tree
    new_tree = C45Tree(splitting_metric="Ratio")
    #create new tree
    new_tree.fit(x, y, a, thresh=0.05)
    new_tree.tree = {"dataset": csv_path} | new_tree.tree
    #save or output tree
    if not save_file_path:
        print(json.dumps(new_tree.tree, indent=2))
    else:
        new_tree.save_tree(save_file_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    else:
        print("Usage: python3 InduceC45.py <csv_path> [<save_file_path>]")