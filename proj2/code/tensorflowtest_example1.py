import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import *

# -------------------------------------------------------------------------------
# 							EXAMPLE 1
# -------------------------------------------------------------------------------

# this example has a 5x5 input, one 3x3 convolution layer with a single kernel, 
# a flatten layer, and a single neuron for the output

# create a feedforward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
# arg for Conv2D: num_kernels, kernel_size, 
# input_shape (input_size, input_size, num_channels)
model.add(layers.Conv2D(1,3,input_shape=(5,5,1), activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# call weight/data generating function
l1, l1b, l2, l2b, input, output = generateExample1()

# set weights to desired values

# first layer weights and bias
l1=l1.reshape(3,3,1,1)
model.layers[0].set_weights([l1,l1b])

# output layer 
model.layers[2].set_weights([np.transpose(l2),l2b])

# setting inputs
img=np.expand_dims(input,axis=(0,3))

#print needed values.
np.set_printoptions(precision=7)
print("-------------------------------------------")
print("-------------------------------------------")
print("		EXAMPLE 1                   ")
print("-------------------------------------------")
print("-------------------------------------------")

print('model output before:')
print(model.predict(img))

print('1st convolutional layer, 1st kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,0]))
print('1st convolutional layer, 1st kernel bias:')
print(np.squeeze(model.get_weights()[1][0]))

print('fully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('fully connected layer bias:')
print(np.squeeze(model.get_weights()[3][0]))


sgd = optimizers.SGD(lr=1)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.fit(img,output,batch_size=10,epochs=100)
print('model output after:')
print(model.predict(img))

print('1st convolutional layer, 1st kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,0]))
print('1st convolutional layer, 1st kernel bias:')
print(np.squeeze(model.get_weights()[1][0]))

print('fully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('fully connected layer bias:')
print(np.squeeze(model.get_weights()[3][0]))
