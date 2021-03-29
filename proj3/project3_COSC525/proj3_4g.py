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

def task4(learning_rate=0.001,momentum=0.9, epochs=1, batch_size=None):
	label_tr = pd.read_csv("fairface_label_train.csv")
	label_te = pd.read_csv("fairface_label_val.csv")

	# Load image dataset 
	x_train, yg_train = load_img_dataset_with_label(label_tr, "gender")
	x_test, yg_test = load_img_dataset_with_label(label_te, "gender")

	yr_train = label_tr['race'].astype('category').cat.codes
	yr_test  = label_te["race"].astype('category').cat.codes

	# Normalize using MinMax Scalar
	x_tr = MinMaxScaling(x_train)
	x_te = MinMaxScaling(x_test)

	# Add another dimension for channel 
	x_te = np.expand_dims(x_te,axis=3)
	x_tr = np.expand_dims(x_tr,axis=3)

	# Label encoding
	encoder = LabelEncoder()
	encoder.fit(yg_train)
	yg_tr = encoder.transform(yg_train)
	yg_te =  encoder.transform(yg_test)

	encoder.fit(yr_train)
	yr_tr = encoder.transform(yr_train)
	yr_te =  encoder.transform(yr_test)

	# Task 4: Your own ConvNet on both tasks simultaneously 
	input = layers.Input(shape=(32,32,1))
	conv1 = layers.Conv2D(filters=32,kernel_size=7,strides=1,
	                    padding='same',activation='relu',
	                    name='conv1')(input)
	max1 = layers.MaxPool2D(pool_size=(3,3),strides=(2,2),
	                        padding="same",name='pool1')(conv1)
	lrn1 = tf.keras.layers.Lambda(
	                    tf.nn.local_response_normalization)(max1)                  
	conv2 = layers.Conv2D(filters=64,kernel_size=(1,1),
	            padding="same",strides=1,activation="relu")(lrn1)
	conv3 = layers.Conv2D(filters=192,kernel_size=(3,3),
	            padding="same",strides=1,activation="relu")(conv2)              
	max2 = layers.MaxPool2D(pool_size=(3,3),strides=(2,2),
	            padding="same")(conv3)
    lrn2 = tf.keras.layers.Lambda(
            tf.nn.local_response_normalization)(max2)    
	fltn = layers.Flatten()(max2)
	fc1_1 = layers.Dense(100, activation='relu')(fltn)
	fc1_2 = layers.Dense(100, activation='relu')(fltn)
	fc2_1 = layers.Dense(2, activation="softmax")(fc1_1)
	fc2_2 = layers.Dense(7, activation="softmax")(fc1_2)

	model = keras.Model(inputs=input,outputs=[fc2_1, fc2_2])

	model.compile(loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], 
	                 optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
	                 metrics=['accuracy'])

	# Fit the model 
	history = model.fit(x=x_tr, y=[yg_tr, yr_tr], 
	                          epochs=epochs,
	                          batch_size=batch_size,
	                          validation_data = (x_te,[yg_te, yr_te]),
	                          verbose=1)

	print(model.summary())

	# Predict using the model 
	pred = model.predict(x_te)
	predicted_class_indices_1=np.argmax(pred[0],axis=1)
	predicted_class_indices_2=np.argmax(pred[1],axis=1)

	labels_1 = {i:x for i,x in enumerate(label_te['gender'].unique())}
	predictions_1 = [labels_1[k] for k in predicted_class_indices_1]

	labels_2 = {i:x for i,x in enumerate(label_te['race'].unique())}
	predictions_2 = [labels_2[k] for k in predicted_class_indices_2]

	# Confusion matrix 
	data_1 = {'y_Actual':    label_te['gender'],
	        'y_Predicted': predictions_1
	        }
	data_2 = {'y_Actual':    label_te['race'],
	        'y_Predicted': predictions_2
	        }

	df_cm_1 = pd.DataFrame(data_1, columns=['y_Actual','y_Predicted'])
	confusion_matrix_1 = pd.crosstab(df_cm_1['y_Actual'], df_cm_1['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
	print(confusion_matrix_1)

	df_cm_2 = pd.DataFrame(data_2, columns=['y_Actual','y_Predicted'])
	confusion_matrix_2 = pd.crosstab(df_cm_2['y_Actual'], df_cm_2['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
	print(confusion_matrix_2)

	# plt.figure(figsize=[12.5,4])
	# plt.subplots(1,2,1)
	sns.heatmap(confusion_matrix_1, annot=True)
	plt.show()
	# plt.subplots(1,2,2, figsize=[5,4])
	sns.heatmap(confusion_matrix_2, annot=True)
	plt.show()

	# Accuracy-vs-epoch and loss-vs-epoch
	plt.figure(figsize=[12.5,4])
	plt.subplot(1,2,1)
	plt.plot(history.history['dense_2_loss'],'-.*', label='train (gender)')
	plt.plot(history.history['val_dense_2_loss'],'-.*', label='val (gender)')
	plt.plot(history.history['dense_3_loss'],'-.*', label='train (race)')
	plt.plot(history.history['val_dense_3_loss'],'-.*', label='val (race)')
	plt.xlabel("Epoch")
	plt.ylabel("Binary Cross Entropy loss")
	plt.title("Task 4: Loss vs epoch (gender and race)")
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(history.history['dense_2_accuracy'],'-.*', label='train (gender)')
	plt.plot(history.history['val_dense_2_accuracy'],'-.*', label='val (gender)')
	plt.plot(history.history['dense_3_accuracy'],'-.*', label='train (race)')
	plt.plot(history.history['val_dense_3_accuracy'],'-.*', label='val (race)')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy (%)")
	plt.title("Task 4: Accuracy vs epoch (gender and race)")
	plt.legend()
	plt.show()