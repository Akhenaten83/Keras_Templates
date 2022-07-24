#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:13:52 2021

@author: danil
"""

# GOOD TUTORIAL HERE: https://www.tensorflow.org/tutorials/keras/keras_tuner

import pandas as pd
import numpy as np

train = pd.read_csv('/Users/train.csv')
test_features = pd.read_csv('/Users/test.csv')
sub = pd.read_csv('/Users/submission_example.csv')
target = 'medv'
train_features = train.drop(target,1)
train_labels = train[target]


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

train.isna().sum()
train.isnull().sum()

train.describe().transpose()[['mean', 'std']]


normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
        ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

tf.random.set_seed(5)

model = build_and_compile_model(normalizer)

history = model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=2, epochs=20)

test_predictions = model.predict(test_features).flatten()
print(model.summary())
print('Dataset name')
print(f'LogLoss:\t\t{log_loss(y_val, val_predictions)}')
print(f'Balanced accuracy:\t{balanced_accuracy_score(val_labels, val_pred_class)}')

