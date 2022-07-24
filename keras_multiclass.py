#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:48:01 2021

@author: danil
"""
from sklearn.metrics import log_loss
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
tf.__version__ # 2.4.1
keras.__version__ # 2.4.0

train = pd.read_csv('/Users/train.csv')
target = 'target'
train.target = train.target-1
train, val = split(train, stratify = train['target'])
train.to_csv('/Users/train_70.csv', index = False )
val.to_csv('/Users/val_30.csv', index = False )

train_features = train.drop(target,1)
train_labels = train[target]
val_features = val.drop('target', 1)
val_labels = val.target

scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
train_labels = np_utils.to_categorical(train_labels)

# normalizer = preprocessing.Normalization()
# normalizer.adapt(np.array(train_features))

def build_and_compile_model():
    model = keras.Sequential([
#        norm,
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(train[target].nunique(), activation = 'softmax')
        ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

tf.random.set_seed(5)

model = build_and_compile_model()

history = model.fit(
    train_features, train_labels,
    batch_size = 64,
    validation_split=0.2,
    verbose=2, epochs=2000)


y_val = encoder.transform(val_labels)
y_val = np_utils.to_categorical(y_val)

#results = model.evaluate(val_features, y_val)

val_predictions = model.predict(val_features)
val_pred_class = [np.argmax(proba) for proba in val_predictions]
print(model.summary())
print('Dataset name')
print(f'LogLoss:\t\t{log_loss(y_val, val_predictions)}')
print(f'Balanced accuracy:\t{balanced_accuracy_score(val_labels, val_pred_class)}')
