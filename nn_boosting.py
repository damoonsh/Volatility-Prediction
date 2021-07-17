import random

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau as RLP
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.losses import mean_absolute_error as MAE
from tensorflow.keras.losses import mean_squared_error as MSE
from tensorflow.keras.optimizers import Adam


layers = [
    Dense(16, activation=None, kernel_initializer=TruncatedNormal(0, 1, 11),
          bias_initializer=TruncatedNormal(1e-1, 1e-3, 11),
          kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None),
    Dense(32, activation='relu', kernel_initializer=TruncatedNormal(0, 2, 11),
          bias_initializer=TruncatedNormal(0, 5, 11), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(64, activation='tanh', kernel_initializer=TruncatedNormal(0, 1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(32, activation=None, kernel_initializer=TruncatedNormal(0, 1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    BatchNormalization(),
    Dense(16, activation=None, kernel_initializer=TruncatedNormal(0, 1, 161),
          bias_initializer=TruncatedNormal(0, 1, 151), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
    Dense(1, activation=None, kernel_initializer=TruncatedNormal(0, 1e-1, 11),
          bias_initializer=TruncatedNormal(0, 1e-1, 32), kernel_regularizer=None,
          bias_regularizer=None, activity_regularizer=None),
]


def nn_rmspe(y_true, y_pred):
    return tf.sqrt(tf.experimental.numpy.nanmean(tf.square(((y_true - y_pred) / y_true))))

class GBNN:

    def __init__(self, x, y, x_test,n: int, layers: [tf.keras.layers], boosting: bool):
        self.X = x
        self.y = y

        self.X_test = x_test

        self.target = y

        self._num_features = x.shape[1]
        self._feature_names = x.columns

        self._n_iterations = n

        self._feature_fractions = [0.5, 0.8, 1]

        self._preds = []

        self._layers = layers

        self._boosting = boosting

        _, self._x_val, _, self._y_val = train_test_split(x, y, test_size=0.2, random_state=2)

    def _base_model(self, input_size: int):
        """ Returns a base model given the input size """
        
        self._layers.insert(0, tf.keras.Input(shape=(input_size,)))

        model = Sequential(self._layers)
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

        # Train the base model
        model = self._base_model(input_size=num_features)
        
        model_hist = model.fit(
            x=sub_x, y=self.target,
            batch_size=256,
            epochs=1000,
            verbose=False,
            shuffle=True,
            validation_data=(self._x_val, self._y_val)
        )

        # Predict and keep the record of it.
        pred = model.predict(sub_x)
        self._preds.append(pred)

        # Update the target for boosting
        if self._boosting: self.target -= pred

    def train(self):
        for i in range(self._n_iterations):
            self._train()
