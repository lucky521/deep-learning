import numpy as np
 
# sigmoid function
# True: f(x) = 1/(1+e^(-x))
# False: f'(x) = f(x)(1-f(x))
def sigmoid(x,deriv=False):
    if(deriv==True): # derivative function: y*(1-y)
        return x*(1-x)
    else:   # origin function
        return 1/(1+np.exp(-x))
 
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
 
y = np.array([[0],
            [1],
            [1],
            [0]])
 
np.random.seed(1)
 
# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
 
    # Feed forward through layers 0, 1, and 2
    l0 = X  # layer 0 - input 
    l1 = sigmoid(np.dot(l0,syn0)) # layer 1 - hidden layer with syn0
    l2 = sigmoid(np.dot(l1,syn1)) # layer 2 - output with syn1

    # how much did we miss the target value?
    l2_error = y - l2
 
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
 
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*sigmoid(l2,deriv=True)
 
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
 
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigmoid(l1,deriv=True)
 
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print 
print "Weights after Training: l1(3X4), l2(4X1)"
print syn0
print syn1
print "Output After Training:"
print l2