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

def task3r(learning_rate=0.001,momentum=0.9, epochs=1, batch_size=None):
    label_tr = pd.read_csv("fairface_label_train.csv")
    label_te = pd.read_csv("fairface_label_val.csv")

    # Load image dataset 
    def load_img_dataset_with_label(df,label):
        x = []
        y = df[label].astype('category').cat.codes
        for i in range(len(df)):
            x.append(image.imread(df['file'][i]))
        return np.array(x, dtype='float32'), np.array(y)
        
    x_train, y_train = load_img_dataset_with_label(label_tr, "race")
    x_test, y_test = load_img_dataset_with_label(label_te, "race")

    # Normalize using MinMax Scalar
    def MinMaxScaling(a):
        return (a - a.min()) / (a.max() - a.min())

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

    # Task 3: Your Own ConvNet (race)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

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
    labels = {i:x for i,x in enumerate(label_te['race'].unique())}
    predictions = [labels[k] for k in predicted_class_indices]

    # Confusion matrix 
    data = {'y_Actual':    label_te['race'],
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
    plt.ylabel("Categorical Cross Entropy loss")
    plt.title("Task 3: Loss vs epoch (race)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],'-.*', label='train')
    plt.plot(history.history['val_accuracy'],'-.*', label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Task 3: Accuracy vs epoch (race)")
    plt.legend()
    plt.show()