import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape, Activation, Dropout
from tensorflow.keras import optimizers

import time 
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 
import numpy as np

from matplotlib import image
from sklearn.preprocessing import LabelEncoder
from proj3 import load_img_dataset_with_label, MinMaxScaling

def task2g(learning_rate=0.001,momentum=0.9, epochs=1, batch_size=None):
	label_tr = pd.read_csv("fairface_label_train.csv")
	label_te = pd.read_csv("fairface_label_val.csv")

	# Load image dataset 
	x_train, y_train = load_img_dataset_with_label(label_tr, "gender")
	x_test, y_test = load_img_dataset_with_label(label_te, "gender")

	# Normalize using MinMax Scalar
	x_tr = MinMaxScaling(x_train)
	x_te = MinMaxScaling(x_test)

	# Add another dimension for channel 
	x_te = np.expand_dims(x_te,axis=3)
	x_tr = np.expand_dims(x_tr,axis=3)

	# Label encoding
	encoder = LabelEncoder()
	encoder.fit(y_train)
	y_tr = encoder.transform(y_train)
	y_te =  encoder.transform(y_test)

	# Build the model 
	model = Sequential()
	model.add(layers.Conv2D(filters=40,kernel_size=5,strides=1,
	                        padding='valid',activation='relu'))
	model.add(layers.MaxPooling2D(pool_size=2))
	model.add(layers.Flatten())
	model.add(layers.Dense(100,activation='relu'))
	model.add(layers.Dense(2,activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy', 
	                 optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
	                 metrics=['accuracy'])

	# Fit the model 
	history = model.fit(x=x_tr, y=y_tr, 
	                          epochs=epochs,
	                          batch_size=batch_size,
	                          validation_data = (x_te,y_te),
	                          verbose=1)

	print(model.summary())

	# Predict using the model 
	pred = model.predict(x_te)
	predicted_class_indices=np.argmax(pred,axis=1)
	predicted_class_indices.shape
	labels = {0: "Female", 1: "Male"}
	predictions = [labels[k] for k in predicted_class_indices]

	# Confusion matrix 
	data = {'y_Actual':    label_te['gender'],
	        'y_Predicted': predictions
	        }

	df_cm = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
	confusion_matrix = pd.crosstab(df_cm['y_Actual'], df_cm['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
	print(confusion_matrix)

	sns.heatmap(confusion_matrix, annot=True)
	plt.show()

	# Accuracy-vs-epoch and loss-vs-epoch
	plt.figure(figsize=[12.5,4])
	plt.subplot(1,2,1)
	plt.plot(history.history['loss'],'-.*', label='train')
	plt.plot(history.history['val_loss'],'-.*', label='val')
	plt.xlabel("Epoch")
	plt.ylabel("Binary Cross Entropy loss")
	plt.title("Task 2: Loss vs epoch (gender)")
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(history.history['accuracy'],'-.*', label='train')
	plt.plot(history.history['val_accuracy'],'-.*', label='val')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy (%)")
	plt.title("Task 2: Accuracy vs epoch (gender)")
	plt.legend()
	plt.show()