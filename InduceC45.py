import pandas as pd
import json
import sys
from c45 import C45Tree

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


#helper function that assists
#in reading csv, properly typing
#all columns, and identifying class
#attribute
def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path)

    #parsing first 3 rows of df
    col_names = pd.Series(df.columns, index=df.columns)
    data_types = df.iloc[0].astype("int64")
    class_var = df.iloc[1, 0]

    rowid_cols = col_names[data_types == -1]
    a = col_names[(col_names != class_var) & (~col_names.isin(rowid_cols))]

    # drop metadata
    df = df.drop([0, 1]).reset_index(drop=True)

    col_types = pd.Series(
        ["float64" if t == 0 else "category" for t in data_types if t != -1],
        index=col_names[data_types != -1]
    ).to_dict()

    df = df.astype(col_types)

    #separate to y and X
    y = df.loc[:, class_var]
    x = df.drop(columns=[class_var, *rowid_cols.tolist()], errors="ignore")
    return x, y, a.reset_index(drop=True)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    else:
        print("Usage: python3 InduceC45.py <csv_path> [<save_file_path>]")