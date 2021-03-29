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
import random

from proj3 import load_img_dataset_with_label, MinMaxScaling

def task5(learning_rate=0.001,momentum=0.9, epochs=1, batch_size=None):
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

     # reparameterization trick
     # instead of sampling from Q(z|X), sample epsilon = N(0,I)
     # z = z_mean + sqrt(var) * epsilon
     from tensorflow.keras import backend as K

     def sampling(args):
         """Reparameterization trick by sampling from an isotropic unit Gaussian.
         # Arguments
             args (tensor): mean and log of variance of Q(z|X)
         # Returns
             z (tensor): sampled latent vector
         """
         #Extract mean and log of variance
         z_mean, z_log_var = args
         #get batch size and length of vector (size of latent space)
         batch = K.shape(z_mean)[0]
         dim = K.int_shape(z_mean)[1]
         
         # by default, random_normal has mean = 0 and std = 1.0
         epsilon = K.random_normal(shape=(batch, dim))
         #Return sampled number (need to raise var to correct power)
         return z_mean + K.exp(z_log_var) * epsilon

     # Task 5: Variational Auto Encoder (VAE)
     # encoder
     latent_dim = 5
     inputs = Input(shape=(32,32,1), name='encoder_input')
     encoder_hl1 = layers.Conv2D(32, kernel_size=7, strides=1, 
                         activation='relu', name='encoder_hl1',
                         padding='same')(inputs)
     encoder_hl2 = layers.Conv2D(64, kernel_size=3, strides=1, 
                         activation='relu', name='encoder_hl2',
                         padding='same')(encoder_hl1)
     encoder_flatten = layers.Flatten()(encoder_hl2)
     z_mean = layers.Dense(latent_dim, name='z_mean')(encoder_flatten)
     z_log_var = layers.Dense(latent_dim, name='z_log_var')(encoder_flatten)

     z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
     encoder = keras.Model(inputs,[z_mean,z_log_var,z], name='encoder_output')
     print(encoder.summary())

     # decoder
     latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
     decoder_dense = layers.Dense(32*32*64, activation='relu')(latent_inputs)
     decoder_reshape = layers.Reshape((32,32,64))(decoder_dense)
     decoder_hl1 = layers.Conv2DTranspose(64, kernel_size=3, strides=1,
                     activation='relu', name='decoder_hl1',
                     padding='same')(decoder_reshape)
     decoder_hl2 = layers.Conv2DTranspose(32, kernel_size=7,strides=1,
                     activation='relu', name='decoder_hl2',
                     padding='same')(decoder_hl1)
     decoder_outputs = layers.Conv2DTranspose(1,kernel_size=1, activation="sigmoid",
                     padding='same')(decoder_hl2)

     decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
     print(decoder.summary())

     outputs = decoder(encoder(inputs)[2])
     vae = keras.Model(inputs, outputs, name='vae_mlp')

     #setting loss
     reconstruction_loss = keras.losses.mse(inputs, outputs)
     reconstruction_loss *=1
     # kl_loss = K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1
     # kl_loss = K.sum(kl_loss, axis=-1)
     # kl_loss *= 0.001
     kl_loss = 0
     vae_loss = K.mean(reconstruction_loss + kl_loss)
     vae.add_loss(vae_loss)
     vae.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum), metrics=['accuracy'])

     # Fit the model 
     history = vae.fit(x=x_tr, y=x_tr, 
                               epochs=epochs,
                               batch_size=batch_size,
                               shuffle=True,
                               verbose=1)

     print(vae.summary())

     # Predict using the model 
     pred = vae.predict(x_te)

     # Plot 10 random latent vectors
     plt.rcParams["axes.grid"] = False

     randomlist = [random.randint(0,len(x_te)) for i in range(10)]

     for i in randomlist:
          plt.imshow(pred[i][:,:,0])
          plt.show()

