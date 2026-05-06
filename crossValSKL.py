import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from InduceC45 import read_csv

def main(csv_file, hyperparams_file, output_tree_file=None):
    x, y, a = read_csv(csv_file)
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    thresholds = hyperparams.get("InfoGain", [])

    # Detect categorical vs numeric columns
    cat_cols = [c for c in x.columns if x[c].dtype.name == "category"]
    num_cols = [c for c in x.columns if x[c].dtype.name == "float64"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    best_accuracy = 0
    best_confusion_matrix = None

    # Create 10-fold split
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform grid search
    for threshold in thresholds:
        accuracies = []
        all_confusion_matrices = []
        all_labels = np.sort(y.unique())

        for train_index, test_index in kf.split(x):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = Pipeline(
                steps=[
                    ("prep", preprocess),
                    ("model", DecisionTreeClassifier(
                        criterion="entropy",
                        min_impurity_decrease=threshold,
                        random_state=42
                    )),
                ]
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracies.append(accuracy_score(y_test, predictions))
            all_confusion_matrices.append(confusion_matrix(y_test, predictions, labels=all_labels))

        overall_accuracy = sum(accuracies) / len(accuracies)
        overall_confusion_matrix = sum(all_confusion_matrices)

        # Track the best model
        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_threshold = threshold
            best_confusion_matrix = overall_confusion_matrix

    print(f"Best Splitting Metric: \nInfoGain \nThreshold: {best_threshold}")
    print("Best Accuracy:", best_accuracy)

    labels = np.sort(y.unique())
    cm_df = pd.DataFrame(best_confusion_matrix, index=labels, columns=labels)
    print(cm_df.to_string())

    # Train on full dataset and save tree if output file is provided
    if output_tree_file:
        final_model = Pipeline(
            steps=[
                ("prep", preprocess),
                ("clf", DecisionTreeClassifier(
                    criterion="entropy",
                    min_impurity_decrease=best_threshold,
                    random_state=42
                )),
            ]
        )
        final_model.fit(x, y)

        clf = final_model.named_steps["clf"]
        feature_names = final_model.named_steps["prep"].get_feature_names_out()

        tree_text = export_text(clf, feature_names=list(feature_names))
        with open(output_tree_file, "w") as f:
            f.write(tree_text)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        main(args[0], args[1])
    elif len(args) == 3:
        main(args[0], args[1], args[2])
    else:
        print("Usage: python3 crossValSKL.py <csv_path> <hyperparams_path> [output_tree_file]")

