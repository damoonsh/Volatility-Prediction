import random

class GBNN:

    def __init__(self, x, y, n: int):
        self.X = x
        self.y = y

        self.target = y

        self._num_features = x.shape[1]
        self._feature_names = x.columns

        self._n_iterations = n
        
        self._feature_fractions = [0.5, 0.8, 1]

        self.preds = []

    def _base_model(self, input_size: int):
        """ returns a base model given the input size """
        pass

    def _train(self):
        # Getting the number of features
        num_features = int(random.choice(self._feature_fractions) * self._num_features)
        # Getting list of features to be used
        feature_names = random.choices(self._feature_names, k=num_features)
        # Subset X based on column
        sub_x = self.X[feature_names]

        # train the base model
        model = self._base_model(input_size=num_features)
        model.fit(sub_x, self.target)

        # Predict and keep the record of it.
        pred = model.predict(sub_x)
        preds.append(pred)

        # update the target
        self.target = y - pred
    
    def train(self):
        for i in range(self._n_iterations):
            self._train()
