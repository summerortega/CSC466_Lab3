class RandomForest:
    def __init__(self, numAttributes, numDataPoints, numTrees, splitting_metric='Ratio', splitting_threshold=0.05):
        self.numTrees = numTrees
        self.numDataPoints = numDataPoints
        self.numAttributes = numAttributes
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.forest = []

        def fit(self, X, y, a):
            #TODO

        def predict(self, X_test):
            #TODO.

