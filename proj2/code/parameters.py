import numpy as np

#Generate data and weights for "example2"
def generateExample2():
    # this example has an input of 7x7, one 3x3 convolution layer
    # with two kernels, another 3x3 convolution layer with a single kernel
    # a flatten layer and a single neuron for the output

    #Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

    #First hidden layer, two kernels
    l1k1=np.random.rand(3,3)
    l1k2=np.random.rand(3,3)
    l1b1=np.random.rand(1)
    l1b2=np.random.rand(1)

    # first layer: 5x5x2=50 neurons, (3x3+1)x2=20 weights 

    #second hidden layer, one kernel, two channels
    l2c1=np.random.rand(3,3)
    l2c2=np.random.rand(3,3)
    l2b=np.random.rand(1)

    # second layer: 3x3x1=9 neurons, (3x3x2+1)x1=19 weights 

    #output layer, fully connected
    l3=np.random.rand(1,9)
    l3b=np.random.rand(1)

    # output layer: 1 neuron, 10 weights 

    #input and output
    input=np.random.rand(7,7)
    output=np.random.rand(1)

    return l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input,output

def generateExample1():
    # this example has a 5x5 input, one 3x3 convolution layer
    # with a single kernel, a flatten layer, 
    # and a single neuron for the output

    # set a seed 
    np.random.seed(42)

    # 1 hidden layer, 1 kernel
    l1  = np.random.rand(3,3)
    l1b = np.random.rand(1)

    # first layer: 3x3x1=9 neurons, (3x3x1+1)x1=10 weights

    # output layer, fully connected
    l2  = np.random.rand(1,9)
    l2b = np.random.rand(1)

    # output layer: 1 neuron, 10 weights 

    # input and output
    input = np.random.rand(5,5)
    output = np.random.rand(1)

    # print("expected output: ", output)
    return l1, l1b, l2, l2b, input, output 

def generateExample3():
    # this example has a 8x8 input, one 3x3 convolution layer
    # with two kernels, a 2x2 max pooling layer, a flatten layer,
    # a single neuron for the output

    # set a seed
    np.random.seed(42)

    # 1 hidden layer, 2 kernels
    l1k1 = np.random.rand(3,3)
    l1k2 = np.random.rand(3,3)
    l1b1 = np.random.rand(1)
    l1b2 = np.random.rand(1)

    # first layer: 6x6x2=72 neurons, (3x3x1+1)x2=20 weights 
    # pooling layer: 3x3x2 neurons

    # output layer, fully connected
    l2   = np.random.rand(1,18)
    l2b  = np.random.rand(1)

    # output layer: 1 neuron, 72 weights 

    # input and output
    input = np.random.rand(8,8)
    output = np.random.rand(1)

    return l1k1, l1k2, l1b1, l1b2, l2, l2b, input, output 


