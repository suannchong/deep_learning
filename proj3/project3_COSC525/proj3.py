import time 
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 
import numpy as np
import sys

from matplotlib import image
# Helper function to load dataset 
def load_img_dataset_with_label(df,label):
    x = []
    y = df[label].astype('category').cat.codes
    for i in range(len(df)):
        x.append(image.imread(df['file'][i]))
    return np.array(x, dtype='float32'), np.array(y)

# Helper function to normalize the dataset
def MinMaxScaling(a):
    return (a - a.min()) / (a.max() - a.min())

if __name__ == "__main__":

     if (len(sys.argv)<2):
          print("usage: python proj3 task[1-5].")

     elif sys.argv[1]=="task1":
          print("-------------------------------------------")
          print("-------------------------------------------")
          print("                  TASK 1                   ")
          print("-------------------------------------------")
          print("-------------------------------------------")

          from proj3_1g import task1g
          from proj3_1r import task1r

          task1g(learning_rate=0.001, momentum=0.9, epochs=50, batch_size=32)
          task1r(learning_rate=0.001, momentum=0.9, epochs=50, batch_size=32)

     elif sys.argv[1]=="task2":
          print("-------------------------------------------")
          print("-------------------------------------------")
          print("                  TASK 2                   ")
          print("-------------------------------------------")
          print("-------------------------------------------")

          from proj3_2g import task2g
          from proj3_2r import task2r

          task2g(learning_rate=0.01, momentum=0.9, epochs=50, batch_size=128)
          task2r(learning_rate=0.01, momentum=0.9, epochs=50, batch_size=128)

     elif sys.argv[1]=="task3":
          print("-------------------------------------------")
          print("-------------------------------------------")
          print("                  TASK 3                   ")
          print("-------------------------------------------")
          print("-------------------------------------------")

          from proj3_3g import task3g
          from proj3_3r import task3r

          task3g(learning_rate=0.01, momentum=0.9, epochs=25, batch_size=256)
          task3r(learning_rate=0.01, momentum=0.9, epochs=25, batch_size=256)

     elif sys.argv[1]=="task4":
          print("-------------------------------------------")
          print("-------------------------------------------")
          print("                  TASK 4                   ")
          print("-------------------------------------------")
          print("-------------------------------------------")

          from proj3_4g import task4

          task4(learning_rate=0.01, momentum=0.9, epochs=25, batch_size=64)

     elif sys.argv[1]=="task5":
          print("-------------------------------------------")
          print("-------------------------------------------")
          print("                  TASK 5                   ")
          print("-------------------------------------------")
          print("-------------------------------------------")

          from proj3_5g import task5

          task5(learning_rate=0.01, momentum=0.9, epochs=25, batch_size=256)

     else:
          print("Please choose task[1-5].")

