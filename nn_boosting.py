import random
import copy

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau as RLP, EarlyStopping as ES
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.losses import mean_absolute_error as MAE
from tensorflow.keras.losses import mean_squared_error as MSE
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

layers = [
    Dense(16, activation=None, kernel_initializer=TruncatedNormal(0, 1, 11),
          bias_initializer=TruncatedNormal(1e-1, 1e-3, 11),
          kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None),
    Dense(32, activation='relu', kernel_initializer=TruncatedNormal(0, 2, 11),
          bias_initializer=TruncatedNormal(0, 5, 11), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(64, activation=None, kernel_initializer=TruncatedNormal(0, 1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    Dense(128, activation='relu', kernel_initializer=TruncatedNormal(0, 1e-1, 61),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(64, activation=None, kernel_initializer=TruncatedNormal(0, 2, 11),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    Dense(32, activation='relu', kernel_initializer=TruncatedNormal(0, 5e-1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    Dense(16, activation=None, kernel_initializer=TruncatedNormal(0, 8e-1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(1, activation=None, kernel_initializer=TruncatedNormal(0, 1e-1, 11),
          bias_initializer=TruncatedNormal(0, 1e-1, 32), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
]


def nn_rmspe(y_true, y_pred):
    return tf.sqrt(tf.experimental.numpy.nanmean(tf.square(((y_true - y_pred) / y_true))))


class GBNN:

    def __init__(self, x, y, x_test, n: int, layers: [tf.keras.layers], boosting: bool):
        self.X = x
        self.X_test = x_test
        self.target = y

        self._num_features = x.shape[1]  # Total number of features
        self._feature_names = x.columns  # All the feature names

        self._n_iterations = n  # Number of base learners to run

        # Different fractions to be randomly selected from
        self._feature_fractions = [0.5, 0.6, 0.8, 0.9, 1]
        self._preds = []  # Keep track of predictions

        self._layers = layers  # layers for the base Neural Network model
        self._boosting = boosting  # If the model should operate based on boosting

        self._average_pred = 0

        self._metrics = {
            "features": [],
            "history": []
        }

    def _base_model(self, input_size: int):
        """ Returns a base model given the input size """

        # Adding the input based on the number of features
        layers = [tf.keras.Input(shape=(input_size,))] + self._layers

        # Instantiating the base model
        # Note: A deep copy of the layers list should be passed!
        model = Sequential(copy.deepcopy(layers))
        model.compile(optimizer=Adam(1e-3), loss=nn_rmspe)

        return model

    def _train(self):
        # Getting the number of features
        num_features = int(random.choice(
            self._feature_fractions) * self._num_features)
        # Getting list of features to be used
        feature_names = random.choices(self._feature_names, k=num_features)
        # Subset X based on column
        sub_x = self.X[feature_names]

        # Getting the validation data to control over fitting
        _, x_val, _, y_val = train_test_split(
            sub_x, self.target, test_size=0.2, random_state=2)

        print(f'---- \n features: {num_features}') # Debuging

        # Train the base model
        model = self._base_model(input_size=num_features)

        model_hist = model.fit(
            x=sub_x, y=self.target,
            batch_size=256,
            epochs=2,
            verbose=True,
            shuffle=True,
            validation_data=(x_val, y_val),
            callbacks=[
                RLP(monitor='val_loss', factor=0.98, patience=15, verbose=1),
                ES(monitor='val_loss', patience=100,
                   verbose=1, restore_best_weights=True)
            ],
        )

        # Predict and keep the record of it.
        pred = model.predict(sub_x)
        self._preds.append(pred)

        self._metrics["features"].append(feature_names)
        self._metrics["history"].append(model_hist.history)

        self._average_pred += pred / self._n_iterations

        # Update the target for boosting
        if self._boosting:
            self.target -= pred

    def train(self):
        for i in range(self._n_iterations):
            self._train()
