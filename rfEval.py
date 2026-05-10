import sys
from sklearn.model_selection import train_test_split

from mlreport import Report, ComparisonReport
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from randomForest import RandomForest
import json

from readcsv import read_csv

def main(csv_file, output_file="evalReport.pdf", grid_file=None):
    #read the data
    x, y, a = read_csv(csv_file)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    #Grid Search
    with open(grid_file, "r") as f:
        grid = json.load(f)

    best_rf = None
    best_params = None
    best_acc = -1

    for m in grid['NumAttributes']:
        for k in grid['PercentData']:
            for n in grid['NumTrees']:
                for thresh in grid.get("threshold", [0.05]):
                    for split in grid.get('split', ["InfoGain"]):
                        rf = RandomForest(
                            num_attributes=m,
                            num_datapoints=k,
                            num_trees=n,
                            splitting_metric=split,
                            splitting_threshold=thresh,
                        )

                        rf.fit(x_train, y_train,a)
                        preds = rf.predict(x_test)
                        acc = (y_test.values == preds).sum() / len(y_test)

                        if acc > best_acc:
                            best_acc = acc
                            best_params = {"NumAttributes": m, "PercentData": k, "NumTrees": n,
                                           "threshold": thresh, "split": split}
                            best_rf = rf

    print("Best RF params:", best_params)
    print("Best RF accuracy:", best_acc)

    # SKlearn Random Forest Grid Search
    cat_cols = [c for c in x.columns if x[c].dtype.name == "category"]
    num_cols = [c for c in x.columns if x[c].dtype.name == "float64"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    best_skl = None
    best_skl_params = None
    best_skl_acc = -1

    for m in grid["NumAttributes"]:
        for k in grid["PercentData"]:
            for n in grid["NumTrees"]:
                clf = Pipeline(
                    steps=[
                        ("prep", preprocess),
                        ("model", RandomForestClassifier(
                            n_estimators=n,
                            max_features=m,
                            bootstrap=True,
                            max_samples=k if (isinstance(k, float) and 0 < k <= 1) else None,
                            random_state=42
                        ))
                    ]
                )

                clf.fit(x_train, y_train)
                preds = clf.predict(x_test)
                acc = (y_test.values == preds).sum() / len(y_test)

                if acc > best_skl_acc:
                    best_skl_acc = acc
                    best_skl_params = {"NumAttributes": m, "PercentData": k, "NumTrees": n}
                    best_skl = clf

    print("Best SKL RF params:", best_skl_params)
    print("Best SKL RF accuracy:", best_skl_acc)

    rf_report = (
        Report(best_rf, title="Custom Random Forest", model_type="classification", model_params=best_params)
        .add_split("train", x_train, y_train)
        .add_split("test", x_test, y_test)
        .build()
    )

    skl_report = (
        Report(best_skl, title="Sklearn Random Forest", model_type="classification", model_params=best_skl_params)
        .add_split("train", x_train, y_train)
        .add_split("test", x_test, y_test)
        .build()
    )

    comp = (
        ComparisonReport(
            reports=[skl_report, rf_report],
            title="Model Comparison",
            split="test",
            theme="light",
        )
        .build()
    )

    comp.to_pdf(output_file)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    elif len(args) == 3:
        main(args[0], args[1], args[2])
    else:
        print("Usage: python3 rfEval.py <CSVFile> [outputfilename] [gridSearchSettings]")


