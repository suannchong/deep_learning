import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import *

# -------------------------------------------------------------------------------
# 							EXAMPLE 3
# -------------------------------------------------------------------------------
# this example has a 8x8 input, one 3x3 convolution layer with two kernels, 
# a 2x2 max pooling layer, a flatten layer, and a single neuron for the output

# create a feedforward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
# arg for Conv2D: num_kernels, kernel_size, 
# input_shape (input_size, input_size, num_channels)
model.add(layers.Conv2D(filters=2,kernel_size=3,input_shape=(8,8,1), activation='sigmoid'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# call weight/data generating function
l1k1, l1k2, l1b1, l1b2, l2, l2b, input, output  = generateExample3()

# set weights to desired values

# first layer weights and bias
l1k1 =l1k1.reshape(3,3,1,1)
l1k2 =l1k2.reshape(3,3,1,1)

w1=np.concatenate((l1k1,l1k2),axis=3)		# concatenate on the third axis (num_kernel)
model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

# output layer 
model.layers[3].set_weights([np.transpose(l2),l2b])

# setting inputs
img=np.expand_dims(input,axis=(0,3))

#print needed values.
np.set_printoptions(precision=5)
print("-------------------------------------------")
print("-------------------------------------------")
print("		EXAMPLE 3                   ")
print("-------------------------------------------")
print("-------------------------------------------")

print('model output before:')
print(model.predict(img))

# print('1st convolutional layer, 1st kernel weights:')
# print(np.squeeze(model.get_weights()[0][:,:,0,0]))
# print('1st convolutional layer, 1st kernel bias:')
# print(np.squeeze(model.get_weights()[1][0]))

# print('1st convolutional layer, 2nd kernel weights:')
# print(np.squeeze(model.get_weights()[0][:,:,0,1]))
# print('1st convolutional layer, 2nd kernel bias:')
# print(np.squeeze(model.get_weights()[1][1]))

# print('fully connected layer weights:')
# print(np.squeeze(model.get_weights()[2]))
# print('fully connected layer bias:')
# print(np.squeeze(model.get_weights()[3][0]))

sgd = optimizers.SGD(lr=1)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.fit(img,output,batch_size=1,epochs=1)

print('model output after:')
print(model.predict(img))

print('1st convolutional layer, 1st kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,0]))
print('1st convolutional layer, 1st kernel bias:')
print(np.squeeze(model.get_weights()[1][0]))

print('1st convolutional layer, 2nd kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,1]))
print('1st convolutional layer, 2nd kernel bias:')
print(np.squeeze(model.get_weights()[1][1]))

print('fully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('fully connected layer bias:')
print(np.squeeze(model.get_weights()[3][0]))

