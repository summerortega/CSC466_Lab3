import sys
import numpy as np
import pandas as pd
from c45 import C45Tree

def main(csv_path:str, load_file_path:str, evalu:str = None) -> None:
    # get training set x, class var y, and attribute series a
    x, y, a = read_csv(csv_path)
    new_tree = C45Tree(splitting_metric="ig")
    new_tree.read_tree(load_file_path)
    results = new_tree.predict(x)
    if not evalu:
        for result in results:
            print(result)
    else:
        print_results(y, np.array(results))


#used if eval parameter of main is set
def print_results(y:pd.Series, results:np.ndarray) -> None:
    print(f"Results: {results}")
    total = len(results)
    print(f"Total Records classified:{total}")
    incorrect = y.compare(pd.Series(results))
    num_correct = len(y) - len(incorrect)
    num_incorrect = len(incorrect)
    print(f"Correct: {num_correct}")
    print(f"Incorrect: {num_incorrect}")
    print(f"Accuracy: {num_correct / total}")
    print("Confusion Matrix:")
    confusion_matrix(y, results)


#create confusion matrix between ground truth and predictions
def confusion_matrix(y:pd.Series, results:np.ndarray) -> None:
    matrix = {}
    actual_values = np.array(y.unique())
    for val in actual_values:
        predictions = results[y == val]
        matrix[val] = {i: len(predictions[predictions == i]) for i in actual_values}
    print("    ", end="")
    for val in matrix.keys():
        print(val, end="  ")
    print('')
    for att in matrix:
        print(f"{att}: {list(matrix[att].values())}")

#helper function that assists
#in reading csv, properly typing
#all columns, and identifying class
#attribute
def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path)
    #parse new df to get types and class var
    col_names = pd.Series(df.columns)
    data_types = df.iloc[0]
    class_var = df.iloc[1, 0]
    df = df.drop([0, 1])
    a = col_names[0: -1]
    data_types = data_types.replace({"0":"float64", r"^[1-9]$":"category"}, regex=True)
    col_types = pd.Series(data_types, index=col_names).to_dict()
    df = df.astype(col_types).reset_index(drop=True)
    #separate to y and X
    y = df.loc[:, class_var]
    x = df.drop(columns=class_var)
    return x, y, pd.Series(a)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        main(args[0], args[1])
    elif len(args) == 3:
        main(args[0], args[1], args[2])
    else:
        print("Usage: python3 predict.py <csv_path> <load_file_path> [eval_option]")
