import pandas as pd
from c45 import C45Tree

class RandomForest:
    #Only info gain needs to be used throughout the tests
    def __init__(self, num_attributes, num_datapoints, num_trees, splitting_metric='InfoGain', splitting_threshold=0.05):
        self.num_trees = num_trees
        self.num_datapoints = num_datapoints
        self.num_attributes = num_attributes
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.forest = []

    def fit(self, x:pd.DataFrame, y:pd.Series, a:pd.Series):
        #For the n number of trees being created
        for _ in range(self.num_trees):
            #sample x datapoints from X and Y using replacement
            x_sample = x.sample(n=self.num_datapoints, replace=True, axis=0)
            y_sample = y.loc[x_sample.index]
            #select y random attributes from the attribute set without replacement
            a_sample = a.sample(n=self.num_attributes, replace=False, axis=0)
            #call the decision tree fit method with the sampled data and new attribute set
            new_tree = C45Tree(splitting_metric=self.splitting_metric, splitting_threshold=self.splitting_threshold)
            new_tree.fit(x_sample, y_sample, a_sample, self.splitting_threshold)
            #add the newly fitted tree to the forest
            self.forest.append(new_tree)

    def predict(self, x_test):
        preds = [tree.predict(x_test) for tree in self.forest]
        results = []
        for i in range(len(x_test)):
            votes = [pred[i] for pred in preds]
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            tied = sorted([label for label, c in counts.items() if c == max_count])
            results.append(tied[0])
        return results