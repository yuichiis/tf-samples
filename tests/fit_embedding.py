# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
from sys import argv

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

print('tensorflow: %s, keras: %s' % (tf.__version__, keras.__version__))

model = keras.models.Sequential([
    keras.layers.Embedding(
        10,
        10,
        input_length=4,
    ),
    keras.layers.Activation('softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'],
)
model.summary()
for w in model.get_weights():
    print(w.shape,np.amax(w))

x = np.array([
    [0,1,2,9],
    [9,8,7,6],
    [1,3,3,4],
    [5,4,3,2],
],np.int32)
t = np.array([
    [0,1,2,9],
    [9,8,7,6],
    [1,3,3,4],
    [5,4,3,2],
])
v_x = np.array([
    [2,3,3,4],
    [1,1,1,4],
    [4,3,3,1],
    [9,3,3,2],
],np.int32)
v_t = np.array([
    [2,3,3,4],
    [1,1,1,4],
    [4,3,3,1],
    [9,3,3,2],
])

print('x=',x.shape)
print('t=',t.shape)

print("training model ...")
history = model.fit(x,t,
    epochs=300,batch_size=1,verbose=0,
    validation_data=(v_x,v_t))

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend()
plt.title('Embedding')
plt.show()
