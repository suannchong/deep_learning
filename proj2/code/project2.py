import numpy as np
import sys
import matplotlib.pyplot as plt 
from parameters import *
# import tensorflow as tf 
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, 
    #learning rate, and possibly with set weights
    def __init__(self,activation, input_size, lr, weights=None):   
        self.activation     = activation
        self.input_size     = input_size       # input size: kw*kh*c+1
        self.lr             = lr
        self.weight_size    = weights.shape
        self.weights        = weights           # kernel: (kw, kh) + 1 
        self.net            = 0                 # input*w for neuron (before activation)
        self.input          = 0                 # input for neuron
        self.output         = 0                 # output for neutron (after activation)
        self.dEdw           = 0                 # partial deriv of weights for neuron

    #This method returns the activation of the net
    def activate(self,net):

        if self.activation == 0:            # linear function
            return net

        return 1/(1+np.exp(-net))           # sigmoid 
        
    #Calculate the output of the neuron should save the input and 
    # output for back-propagation.   
    def calculate(self,input):
        self.input = input
        self.net = np.matmul(self.input,self.weights)
        self.output = self.activate(self.net)

        return self.output

    #This method returns the derivative of the activation function 
    #with respect to the net   
    def activationderivative(self):

        if self.activation == 0:            # linear function
            return 1

        activationderiv = np.exp(-self.net)/(1+np.exp(-self.net))**2  
        return activationderiv
    
    #This method calculates the partial derivative for each weight 
    #and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):

        delta = np.array(wtimesdelta*self.activationderivative())
        self.dEdw = delta*self.input
        wtimesdeltas = delta*self.weights

        return wtimesdeltas, self.dEdw
    
    #Simply update the weights using the partial derivatives and 
    # the learning weight
    def updateweight(self, dEdw=None):
        if dEdw is not None:
            self.dEdw = dEdw 

        self.weights = self.weights - self.lr*self.dEdw
        return self.weights

class ConvolutionalLayer:
    # initialize with the number of neurons in the layer, their activation,
    # the input size, the learning rate and a 2d matrix of weights 
    # (or else initilize randomly)
    def __init__(self, num_kernels, kernel_size, activation, input_size, lr, weights):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size              # assume square kernel size
        self.activation = activation
        self.input_size = input_size                # assume (w,h,c) = (width, height, num_channels)
        self.lr         = lr
        self.stride     = 1                         # assume stride = 1
        self.offset     = 0                         # assume padding = 'valid'
        self.weights    = weights
        
        # if no weights given, randomly initialize the weights 
        # weights shape = (k, 2) where [k][0] is the weights (kw, kh, c), [k][1] is the bias 
        if weights is None:
            self.weights = np.array([[np.random.rand(self.kernel_size, self.kernel_size, self.input_size[2]), np.random.rand(1)] 
                for i in range(self.num_kernels)], dtype=object)

        # calculate the number of neurons in the layer 
        ow   = int((self.input_size[0]- self.kernel_size)/self.stride + 1)
        oh   = int((self.input_size[1]- self.kernel_size)/self.stride + 1)
        oc   = self.num_kernels

        self.numOfNeurons = ow*oh*oc
        self.output_size = (ow,oh,oc)
        self.output = np.zeros(self.numOfNeurons).reshape(self.output_size)

        # input size for each neuron (w, h, c) = c * w * h + 1 (add 1 for bias node)
        # assume neuron accepts a vector 
        self.neuron_input_num = self.input_size[2]*self.kernel_size*self.kernel_size + 1 
        self.neuron_input_size = (self.kernel_size,self.kernel_size,self.input_size[2])
        
        # create an array of neurons and allocate the right weights to the neurons 
        # output size for neuron array (ow, oh, oc)
        self.Neurons = np.zeros(self.output_size,dtype=object)
        for k in range(self.num_kernels):
            for ow in range(self.output_size[0]):
                for oh in range(self.output_size[1]):
                    # shared weights for neurons in all channels for each kernel: kw * kh * c + 1 
                    neuron_shared_weights = np.append(weights[k][0].flatten(), weights[k][1])
                    self.neuron_input_num = neuron_shared_weights.shape[0]

                    self.Neurons[ow,oh,k] = Neuron(self.activation, self.neuron_input_num, self.lr, neuron_shared_weights)

    # calculate the output of all the neurons in the layer and 
    # return a vector with those values (go through the neurons 
    # and call the calculate() method)      
    def calculate(self, input):
        # input dimension for each conv layer (c,w,h) 
        self.input = input
        # assert self.input_size == input.shape 

        for c in range(self.output_size[2]):
            for w in range(self.output_size[0]):
                for h in range(self.output_size[1]):
                    # need to partition the inputs for each output neuron: c*kw*kh+1
                    # don't need to reshape input size back to original dimension 
                    if (w + self.kernel_size) <= self.input_size[0] and (h + self.kernel_size) <= self.input_size[1]:
                        neuron_input = self.input[w:w+self.kernel_size, h:h+self.kernel_size,:].flatten()
                        neuron_input = np.append(neuron_input, 1)
                        self.output[w,h,c] = self.Neurons[w,h,c].calculate(neuron_input)

        return self.output 
            
    # given the next layer's w*delta, should run through the neurons 
    # calling calcpartialderivative() for each (with the correct value), 
    # sum up its ownw*delta, and then update the weights s
    # (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        # same dimension as the kernel size 
        wtimesdeltas = np.zeros(self.input_size)  
        dEdw = np.zeros(self.neuron_input_num)

        for oc in range(self.output_size[2]):
            for ow in range(self.output_size[0]):
                for oh in range(self.output_size[1]):

                    # for each neuron in the layer, calculate the partial derivatice 
                    neuron_wtimesdelta, neuron_dEdw = self.Neurons[ow,oh,oc].calcpartialderivative(wtimesdelta[ow,oh,oc])
                    dEdw += neuron_dEdw
                    
                    # remove bias node and reshape the neuron_wtimesdelta to (c, kw, kh)
                    neuron_wtimesdelta = neuron_wtimesdelta[:-1].reshape(self.neuron_input_size)

                    for c in range(neuron_wtimesdelta.shape[2]):
                        for kw in range(neuron_wtimesdelta.shape[0]):
                            for kh in range(neuron_wtimesdelta.shape[1]):
                                if (ow + kw) <= self.input_size[0] and (oh + kh) <= self.input_size[1]: 
                                    wtimesdeltas[ow+kw,oh+kh,c] += neuron_wtimesdelta[kw,kh,c]

        self.weights = []
        # sum all the dEdw from all output element-wise then update the weights
        for oc in range(self.output_size[2]):
            for ow in range(self.output_size[0]):
                for oh in range(self.output_size[1]):
                    new_weights = self.Neurons[ow,oh,oc].updateweight(dEdw)
            self.weights.append([new_weights[:-1].reshape(self.kernel_size, self.kernel_size, self.input_size[2]), new_weights[-1]])

        return wtimesdeltas 


class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,
    # the input size, the learning rate and a 2d matrix of weights 
    # (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_size, lr, weights=None):
        self.numOfNeurons = numOfNeurons        # an int (only accepts a vector)
        self.activation = activation
        self.input_size = input_size + 1        # input dimension (w,h) + bias
        self.lr         = lr
        self.weights    = weights 
        self.input      = 0

        # assert self.weights.shape = (self.input_size,self.numOfNeurons)

        # randomly initialized weights if not given 
        if self.weights is None:
            self.weights = np.random.rand(self.numOfNeurons, self.input_size)

        self.Neurons = []
        for i in range(self.numOfNeurons):
            self.Neurons.append(Neuron(self.activation,self.input_size,self.lr,self.weights[i]))

        self.Neurons = np.array(self.Neurons, dtype=object)

    #calculate the output of all the neurons in the layer and 
    # return a vector with those values (go through the neurons 
    # and call the calculate() method)      
    def calculate(self, input):
        self.input = np.append(input,1)
        results = []
        for neuron in self.Neurons:
            results.append(neuron.calculate(self.input))
        
        self.output = np.array(results).reshape(self.Neurons.shape)
        return self.output
            
    # given the next layer's w*delta, should run through the neurons 
    # calling calcpartialderivative() for each (with the correct value), 
    # sum up its ownw*delta, and then update the weights s
    # (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        wtimesdeltas = []

        for i in range(self.numOfNeurons):
            wtimesdeltas.append(self.Neurons[i].calcpartialderivative(wtimesdelta[i])[0])
            self.weights = self.Neurons[i].updateweight()

        return np.sum(wtimesdeltas,axis=0)[:-1]


class MaxPoolingLayer:
    # restrict the layer to 2d max pooling
    # initialize with the size of the kernel (assume square)
    # dimension of the inputs 
    # assume stride is the same as the kernel size with no padding
    def __init__(self, kernel_size, input_size):
        self.kernel_size = kernel_size
        self.stride     = kernel_size                           # stride = kernel_size 
        self.input_size = input_size                            # (c,w,h), not always w = h
        self.mask       = np.zeros(self.input_size)       # keep tracking of max position
        
        # Calculate the number of neurons in the next layer 
        w               = int((self.input_size[1]-self.kernel_size)/self.stride + 1)
        h               = int((self.input_size[2]-self.kernel_size)/self.stride + 1)
        c               = self.input_size[0]
        self.output_size = c,w,h
        self.output     = np.zeros(self.output_size)

        # revert wtimesdelta from reduced size to original input size 
        self.wtimesdeltas = np.zeros(self.input_size)

    # Given an input, calculate the output of the layer 
    def calculate(self,input):

        for c in range(self.input_size[0]):
            for w in range(0,self.input_size[1],self.stride):
                for h in range(0,self.input_size[2],self.stride):
                    if (w + self.kernel_size) <= self.input_size[1] and (h + self.kernel_size) <= self.input_size[2]:
                        pool = input[c,w:w+self.stride,h:h+self.stride]  # do maxpooling by channel
                        largest = pool.max()
                        self.output[c, int(w/self.stride),int(h/self.stride)] = largest # set output to the right values

                        for kw in range(self.kernel_size):
                            for kh in range(self.kernel_size):
                                if pool[kw,kh] == largest:
                                    self.mask[c,w+kw,h+kh] = 1            # keep track of max position

        return self.output
        
    # Given the ∑ w x ∂ from the next layer, return a new ∑ w x ∂
    # which is the size of the input and the values are set in the correct locations 
    def calcwdeltas(self,wtimesdelta):

        for c in range(self.input_size[0]):
            for w in range(self.input_size[1]):
                for h in range(self.input_size[2]):
                    self.wtimesdeltas[c,w,h] = self.mask[c,w,h]*wtimesdelta[c, int(w/self.stride), int(h/self.stride)] 

        return self.wtimesdeltas

class FlattenLayer:
    # initialize with the input size
    def __init__(self, input_size):
        self.input_size = input_size    # dimension of input e.g. (w,h,c)
        self.output_size = input_size[0]*input_size[1]*input_size[2]

    # Given an input, simply resize it to create the output of the layer 
    def calculate(self,input):

        return input.flatten()          # dimension of output e.g. (w*h*c) 

    # Given the ∑ w x ∂ from the next layer, simply resize it to the size of input
    def calcwdeltas(self, wtimesdelta):

        wtimesdeltas = wtimesdelta.reshape(self.input_size)

        return wtimesdeltas
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the input size, activation (for each layer), 
    # the loss function, the learning rate 
    def __init__(self, input_size, loss, lr):
        self.input_size  = input_size             # input size of the first layer 
        self.activation = 1                     # activation function type (default: sigmoid)
        self.loss       = loss                  # loss function type
        self.lr         = lr                    # learning rate
        self.Layers     = []                    # list of layers within neural network 

    # initialize a layer with number of kernels, kernel size, activation
    # and weights (randomly initialize if not specified)
    def addLayer(self, layer_type, numOfNeurons=None, num_kernels=None, kernel_size=None, activation=None, input_size=None, lr=None, weights=None):

        # input_size dimension (w, h, c)
        if input_size is None:
            if len(self.Layers) == 0:
                input_size = self.input_size                # get input size from initialization 
            else:
                input_size = self.Layers[-1].output_size    # get the input size from the current final layer 

        if activation is not None:
            self.activation = activation

        if lr is not None:
            self.lr = lr 

        # Check for different types of layers 
        if layer_type == "Conv2D":
            assert (num_kernels is not None)  and (kernel_size is not None)
            self.Layers.append(ConvolutionalLayer(num_kernels, kernel_size, self.activation, input_size, self.lr, weights))
        elif layer_type == "Dense":
            assert (numOfNeurons is not None)
            self.Layers.append(FullyConnected(numOfNeurons, self.activation, input_size, self.lr, weights))
        elif layer_type == "MaxPooling2D":
            assert (kernel_size is not None)
            self.Layers.append(MaxPoolingLayer(kernel_size, input_size))
        elif layer_type == "Flatten":
            self.Layers.append(FlattenLayer(input_size))
        else:
            print("Please choose one option [Conv2D | Dense| MaxPooling2D | Flatten] ")


    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        results = self.Layers[0].calculate(input)

        for i in range(1,len(self.Layers)):
            results = self.Layers[i].calculate(results)

        return results

        
    #Given a predicted output and ground truth output simply 
    # return the loss (depending on the loss function)
    def calculateloss(self,y,yp):
        if self.loss == 0:
            # return tf.keras.losses.mean_squared_error(y,yp)
            return np.mean((y-yp)**2)
        
        res = []
        for k in range(len(y)):
            res.append([-(i*np.log2(j)+(1-i)*np.log2(1-j)) for i,j in zip(y[k],yp[k])])

        return np.mean(res)
    
    #Given a predicted output and ground truth output simply 
    # return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,y,yp):
        if self.loss == 0:
            return -2*(y-yp)

        return [-i/j + (1-i)/(1-j) for i,j in zip(y,yp)]
    
    #Given a single input and desired output preform one step of backpropagation 
    #(including a forward pass, getting the derivative of the loss, 
    #and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        # feedforward 
        yp = self.calculate(x)

        # backpropagate 
        wtimesdelta = self.Layers[-1].calcwdeltas(self.lossderiv(y,yp))

        for i in range(len(self.Layers)-2,-1,-1):
            wtimesdelta = self.Layers[i].calcwdeltas(wtimesdelta)

if __name__=="__main__":

    if (len(sys.argv)<2):
        print('usage: python project1_suann.py [example1|example2|example3]')
        
    elif(sys.argv[1]=='example1'):
        # this example has a 5x5 input, one 3x3 convolution layer with a single kernel, 
        # a flatten layer, and a single neuron for the output

        print("-------------------------------------------")
        print("-------------------------------------------")
        print("     EXAMPLE 1                   ")
        print("-------------------------------------------")
        print("-------------------------------------------")

        #print needed values.
        np.set_printoptions(precision=5)

        # call weight/data generating function
        l1, l1b, l2, l2b, input, output = generateExample1()
        # print(input)
        
        loss        = 0
        lr          = 1

        # reformat input into (w,h,c)
        input = np.expand_dims(input,axis=2)  
        input_size  = input.shape

        # initialize a neural network 
        model = NeuralNetwork(input_size, loss, lr)

        # First layer: one 3x3 kernel, sigmoid 
        l1 = l1.reshape(3,3,1)   # (kw,kh,c)
        w1 = np.array([[l1,l1b]], dtype=object)       # weights dimension: (k, 2) where [k][0] = (kw, kh, c), k[1] = (1)
        model.addLayer(layer_type="Conv2D", numOfNeurons=None, num_kernels=1, kernel_size=3, 
                        activation=1, input_size=input_size, lr=lr, weights=w1)

        # Flatten layer 
        model.addLayer(layer_type="Flatten")

        # Dense layer 
        w2 =  np.array([np.append(l2.flatten(),l2b)], dtype=object)
        model.addLayer(layer_type="Dense", numOfNeurons=1, activation=1, weights=w2)


        print("model output before: ")
        print(model.calculate(input))

        # training the model
        model.train(input,output)
        yp = model.calculate(input)
        loss = model.calculateloss(output,yp)

        print("model output after")
        print(yp)
        print("loss: ", loss)
        
        print('1st convolutional layer, 1st kernel weights:')
        print(model.Layers[0].weights[0][0][:,:,0])
        print('1st convolutional layer, 1st kernel bias:')
        print(model.Layers[0].weights[0][1])

        print('fully connected layer weights:')
        print(model.Layers[2].weights[:-1])
        print('fully connected layer bias:')
        print(model.Layers[2].weights[-1])

    elif (sys.argv[1]=='example2'):
        # this example has an input of 7x7, one 3x3 convolution layer
        # with two kernels, another 3x3 convolution layer with a single kernel
        # a flatten layer and a single neuron for the output
        print("-------------------------------------------")
        print("-------------------------------------------")
        print("     EXAMPLE 2                   ")
        print("-------------------------------------------")
        print("-------------------------------------------")

        #print needed values.
        np.set_printoptions(precision=5)

        # Call weight/data generating function
        l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample2()
        # print(input)
        
        loss        = 0
        lr          = 1

        # reformat input into (w,h,c)
        input = np.expand_dims(input,axis=2)  
        input_size  = input.shape

        # initialize a neural network 
        model = NeuralNetwork(input_size, loss, lr)

        # First layer: two 3x3 kernel, sigmoid 
        l1k1 = l1k1.reshape(3,3,1)   # (kw,kh,c)
        l1k2 = l1k2.reshape(3,3,1)   # (kw,kh,c)
        w1 = np.array([[l1k1,l1b1],[l1k2, l1b2]], dtype=object)       # weights dimension: (k, 2) where [k][0] = (kw, kh, c), k[1] = (1)
        model.addLayer(layer_type="Conv2D", numOfNeurons=None, num_kernels=2, kernel_size=3, 
                        activation=1, input_size=input_size, lr=lr, weights=w1)
        
        # Second layer: one 3x3 kernel, sigmoid 
        l2c1 = l2c1.reshape(3,3,1)
        l2c2 = l2c2.reshape(3,3,1)
        l2   = np.concatenate((l2c1,l2c2),axis=2)
        w2 = np.array([[l2, l2b]], dtype=object)
        model.addLayer(layer_type="Conv2D", numOfNeurons=None, num_kernels=1, kernel_size=3, 
                        activation=1, lr=lr, weights=w2)
        
        # Flatten layer 
        model.addLayer(layer_type="Flatten")

        # Dense layer 
        w3 =  np.array([np.append(l3.flatten(),l3b)], dtype=object)
        model.addLayer(layer_type="Dense", numOfNeurons=1, activation=1, weights=w3)


        print("model output before: ")
        print(model.calculate(input))

        # training the model
        model.train(input,output)
        yp = model.calculate(input)
        loss = model.calculateloss(output,yp)

        print("model output after")
        print(yp)
        print("loss: ", loss)
        
        print('1st convolutional layer, 1st kernel weights:')
        print(model.Layers[0].weights[0][:-1][0][:,:,0])
        print('1st convolutional layer, 1st kernel bias:')
        print(model.Layers[0].weights[0][-1])

        print('1st convolutional layer, 2nd kernel weights:')
        print(model.Layers[0].weights[1][:-1][0][:,:,0])
        print('1st convolutional layer, 2nd kernel bias:')
        print(model.Layers[0].weights[1][-1])

        print('2nd convolutional layer weights:')
        print(model.Layers[1].weights[0][:-1][0][:,:,0])
        print(model.Layers[1].weights[0][:-1][0][:,:,1])
        print('2nd convolutional layer bias:')
        print(model.Layers[1].weights[0][-1])

        print('fully connected layer weights:')
        print(model.Layers[3].weights[:-1])
        print('fully connected layer bias:')
        print(model.Layers[3].weights[-1])

    elif (sys.argv[1]=='example3'):
        # this example has a 8x8 input, one 3x3 convolution layer with two kernels, 
        # a 2x2 max pooling layer, a flatten layer, and a single neuron for the output
        print("-------------------------------------------")
        print("-------------------------------------------")
        print("     EXAMPLE 3                   ")
        print("-------------------------------------------")
        print("-------------------------------------------")

        #print needed values.
        np.set_printoptions(precision=5)

        # call weight/data generating function
        l1k1, l1k2, l1b1, l1b2, l2, l2b, input, output  = generateExample3()
        
        loss        = 0
        lr          = 1

        # reformat input into (w,h,c)
        input = np.expand_dims(input,axis=2)  
        input_size  = input.shape

        # initialize a neural network 
        model = NeuralNetwork(input_size, loss, lr)

        # First layer: two 3x3 kernel, sigmoid 
        l1k1 = l1k1.reshape(3,3,1)   # (kw,kh,c)
        l1k2 = l1k2.reshape(3,3,1)   # (kw,kh,c)
        w1 = np.array([[l1k1,l1b1],[l1k2, l1b2]], dtype=object)       # weights dimension: (k, 2) where [k][0] = (kw, kh, c), k[1] = (1)
        model.addLayer(layer_type="Conv2D", numOfNeurons=None, num_kernels=2, kernel_size=3, 
                        activation=1, input_size=input_size, lr=lr, weights=w1)

        # MaxPooling layer 
        model.addLayer(layer_type="MaxPooling2D",kernel_size=2)

        # Flatten layer 
        model.addLayer(layer_type="Flatten")

        # Dense layer 
        w2 =  np.array([np.append(l2.flatten(),l2b)], dtype=object)
        model.addLayer(layer_type="Dense", numOfNeurons=1, activation=1, weights=w2)

        print("model output before: ")
        print(model.calculate(input))

        # training the model
        model.train(input,output)
        yp = model.calculate(input)
        loss = model.calculateloss(output,yp)

        print("model output after")
        print(yp)
        print("loss: ", loss)
        
        print('1st convolutional layer, 1st kernel weights:')
        print(model.Layers[0].weights[0][:-1][0][:,:,0])
        print('1st convolutional layer, 1st kernel bias:')
        print(model.Layers[0].weights[0][-1])

        print('1st convolutional layer, 2nd kernel weights:')
        print(model.Layers[0].weights[1][:-1][0][:,:,0])
        print('1st convolutional layer, 2nd kernel bias:')
        print(model.Layers[0].weights[1][-1])

        print('fully connected layer weights:')
        print(model.Layers[3].weights[:-1])
        print('fully connected layer bias:')
        print(model.Layers[3].weights[-1])
    else:
        print("Please choose either example1, example2 or example3 only.")


        