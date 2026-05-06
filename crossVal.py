import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from c45 import C45Tree
from InduceC45 import read_csv
def main(csv_file, hyperparams_file, output_tree_file=None):
    x, y, a = read_csv(csv_file)
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    best_accuracy = 0
    best_hyperparams = None
    best_confusion_matrix = None

    # Create 10-fold split
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform grid search
    for metric in ['InfoGain', 'Ratio']:
        for threshold in hyperparams[metric]:
            accuracies = []
            all_confusion_matrices = []
            all_labels = np.sort(y.unique())

            for train_index, test_index in kf.split(x):
                X_train, X_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train and predict using C45Tree
                model = C45Tree(splitting_metric=metric, splitting_threshold=threshold)
                model.fit(X_train, y_train, a, threshold)
                predictions = model.predict(X_test)

                # Evaluate
                acc = accuracy_score(y_test, predictions)
                accuracies.append(acc)
                all_confusion_matrices.append(confusion_matrix(y_test, predictions, labels=all_labels))

            # Compute overall accuracy and confusion matrix
            overall_accuracy = sum(accuracies) / len(accuracies)
            overall_confusion_matrix = sum(all_confusion_matrices)

            # Track the best model
            if overall_accuracy > best_accuracy:
                best_accuracy = overall_accuracy
                best_hyperparams = (metric, threshold)
                best_confusion_matrix = overall_confusion_matrix

    # Output results
    print(f"Best Splitting Metric: \n{best_hyperparams[0]} \nThreshold: {best_hyperparams[1]}")
    print("Best Accuracy:", best_accuracy)

    labels = np.sort(y.unique())
    cm_df = pd.DataFrame(best_confusion_matrix, index=labels, columns=labels)
    print(cm_df.to_string())

    # Train on full dataset and save tree if output file is provided
    if output_tree_file:
        best_metric, best_threshold = best_hyperparams
        final_model = C45Tree(splitting_metric=best_metric, splitting_threshold=best_threshold)
        final_model.fit(x, y, a, best_threshold)
        final_model.save_tree(output_tree_file)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        main(args[0], args[1])
    elif len(args) == 3:
        main(args[0], args[1], args[2])
    else:
        print("Usage: python3 crossVal.py <csv_path> <hyperparams_path> [output_tree_file]")