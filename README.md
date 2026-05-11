# CSC466_Lab2

Summer Mariana Ortega
sorteg16@calpoly.edu

Diego Melgoza
drmelgoz@calpoly.edu

## Grid Search Parameters

`rfEval.py` takes a JSON grid search file as its third argument:

```bash
python3 rfEval.py <CSVFile> <OutputPDF> <gridSearchParams>
```

The grid file contains lists of values for each parameter. The program tries each combination and keeps the best model.

- `threshold`: C4.5 stopping threshold
- `split`: C4.5 splitting metric
- `NumTrees`: number of trees in the random forest
- `NumAttributes`: number of randomly selected attributes per tree
- `PercentData`: fraction of training data sampled per tree

## Iris Parameters

For Iris, we used a larger grid because the dataset is small. The grid was approximately:

```json
{
  "threshold": [0.01, 0.04, 0.05, 0.1],
  "split": ["Information Gain", "Information Gain Ratio"],
  "NumTrees": [50, 100, 200, 500, 1000],
  "NumAttributes": [2, 3, 4],
  "PercentData": [0.1, 0.25, 0.33, 0.5]
}
```
## Letter Recognition Parameters

For Letter Recognition, we used the grid shown below:

```json
{
  "threshold": [0.01],
  "split": ["InfoGain", "Ratio"],
  "NumTrees": [25, 50, 100],
  "NumAttributes": [4, 6, 8],
  "PercentData": [0.05, 0.1]
}
```

This grid was reduced compared to Iris because Letter Recognition is much larger.
## Heart Disease Parameters

For the Heart Disease dataset, we used the grid shown below:

```json
{
  "threshold": [0.5],
  "split": ["InfoGain"],
  "NumTrees": [5, 10, 20],
  "NumAttributes": [2, 3],
  "PercentData": [0.25, 0.50, 0.75]
}
```

We used a reduced grid for Heart Disease because lower threshold values caused the custom Random Forest implementation to take a very long time. In some runs, the same parameter settings completed quickly, while in other runs they were noticeably slower. We believe this variation was caused by the random sampling of data points and attributes for each tree. Depending on which numeric attributes were selected, the C4.5 split search could become much more expensive because numeric attributes may have many possible split points.
