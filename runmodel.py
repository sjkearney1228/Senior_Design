# -*- coding: utf-8 -*-
"""RunModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UpJ-jEAzRAlymcghxqy9MknSNk90vsPa
"""

import h5py
import numpy as np
import keras
from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#load dataset
X_test1 = []
Y_test1 = []
filename = 'vggNet_binary_1.hdf5'
data = h5py.File(filename, "r")
print('Selected File: '+str(filename))
X_test1.append(np.array(data["test_img"]))
Y_test1.append(np.array(data["test_labels"]))
data.close()
X_test1 = np.concatenate(X_test1,axis=0)
Y_test1 = np.concatenate(Y_test1,axis=0)
print(X_test1.shape)
print(Y_test1.shape)
#Y_test1bin = to_categorical(np.argmax(Y_test1,-1)!=0) # binary
#print(Y_test1bin.shape)

from keras import optimizers
LEARNING_RATE = 1e-4

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=LEARNING_RATE), metrics=['categorical_accuracy'])

preds = loaded_model.evaluate(X_test1, Y_test1)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))