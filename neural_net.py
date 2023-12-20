import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
"""
IGNORE THIS FILE
"""

class Value:
    
    def __init__(self, data=None, label="", _parents=set(), _operation=""):
        """
        Constructor for a value node that holds data + gradient + how the value was produced
        """
        
        # initialize data randomly between -1 and 1 if no data was provided
        self.data = random.uniform(-1, 1) if data is None else data
            
        # initialize a gradient and label used for printing the value
        self.grad = 0
        self.label = label
        
        # initialize how the value was created (what operation using what inputs)
        # initialize how the backward pass for this operation executes
        self._parents = set(_parents)
        self._operation = _operation
        self._backward = lambda: None
    
    def __repr__(self):
        """
        Helper class used for printing the Value
        """
        return f"Value(data={self.data}, label={self.label})"
  
    def __add__(self, other):
        """
        Implementing the + operator
        """
        
        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _parents=(self, other), _operation='+')
    
        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward
    
        return out
    
    def __mul__(self, other):
        """
        Implementing the * operator
        """
        
        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _parents=(self, other), _operation='*')
    
        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
      
        return out

                
    def __pow__(self, other):
        """
        Implementation of the power operator **
        """
        
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, _parents=(self,), _operation="**" + str(other))
        
        def _backward():
            self.grad += (other * (self.data**(other-1))) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        """
        Implementation of ReLU activation function
        """
        
        out = Value(0 if self.data < 0 else self.data, _parents=(self,), _operation=("ReLU"))
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        """
        Calls the _backward() method for each node in reverse topological order
        """

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
    
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def exp(self):
        """
        Implementing exponent
        """
        # TODO implement exp method
        out = Value(np.exp(self.data), _parents=(self,), _operation=("exp"))

        def _backward():
            self.grad += (np.exp(self.data)) * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """
        Implementing log
        """
        
        # TODO implement log method
        out = Value(np.log(self.data), _parents = (self,), _operation = ("log"))

        def _backward():
            self.grad += (1/self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def __float__(self): return float(self.data)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1



def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if y == yhat: acc += 1

    return acc/len(Y) * 100


def sigmoid(value, scale=0.5):
    """
    Returns sig(value) using the scale parameter
    """
    # TODO implement sigmoid
    return 1/(1+(-scale*value).exp())

def negative_loglikelihood(y, pY1):
    """
    Return negative loglikelihood for a single example based on the value of Y and p(Y=1 | ...)
    """

    # TODO implement loglikelihood
    return -(y*pY1.log() + (1-y)*((1- pY1).log()))


class Neuron:
    """
    Class that represents a single neuron in a neural network.
    A neuron first computes a linear combination of its inputs + an intercept,
    and then (potentially) passes it through a non-linear activation function
    """
  
    def __init__(self, n_inputs):
        """
        Constructor which initializes a parameter for each input, and one parameter for an intercept
        """
        self.theta = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.intercept = Value(random.uniform(-1, 1))

    def __call__(self, x, relu=False, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """
        
        # produce linear combination of inputs + intercept
        # TODO edit to implement dropout
        d = np.zeros(len(self.theta))
        for i in range(len(self.theta)):
             if random.uniform(0, 1) > dropout_proba:
                d[i] = 1
        

        if train_mode:
            out = sum([self.theta[i]*x[i]*d[i] for i in range(len(self.theta))]) + self.intercept

        else:
            out = sum([(1-dropout_proba)*self.theta[i]*x[i] for i in range(len(self.theta))]) + self.intercept
        
        # activate using ReLU based on boolean flag
        if relu:
            return out.relu()
        return sigmoid(out)
    
    def parameters(self):
        """
        Method that returns a list of all parameters of the neuron
        """
        return self.theta + [self.intercept]
    
class Layer:
    """
    Class for implementing a single layer of a neural network
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Constructor initializes the layer with neurons that ta
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
  
    def __call__(self, x, relu=True, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """
        
        # produces a list of outputs for each neuron
        outputs = [n(x, relu, dropout_proba, train_mode=train_mode) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
  
    def parameters(self):
        """
        Method that returns a list of all parameters of the layer
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    """
    Class for implementing a multilayer perceptron
    """
  
    def __init__(self, n_features, layer_sizes, learning_rate=0.01, dropout_proba=0.1):
        """
        Constructor that initializes layers of appropriate width
        """

        layer_sizes = [n_features] + layer_sizes
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.dropout_proba = dropout_proba
        self.learning_rate = learning_rate
        # boolean flag that determines whether we are currently training or testing
        # helpful for controlling how dropout is used
        self.train_mode = True
  
    def __call__(self, x):
        """
        Impelementing the call () operator which simply calls each layer in the net
        sequentially using outputs of previous layers
        """
        
        # use ReLU activation for all layers except the last one
        out = x
        for layer in self.layers[0:len(self.layers)-1]:
            out = layer(out, relu=True, dropout_proba =self.dropout_proba, train_mode=self.train_mode)
        return self.layers[-1](out, relu=False)
  
    def parameters(self):
        """
        Method that returns a list of all parameters of the neural network
        """
        return [p for layer in self.layers for p in layer.parameters()]
    
    def _zero_grad(self):
        """
        Method that sets the gradients of all parameters to zero
        """
        for p in self.parameters():
            p.grad = 0
            
    def fit(self, Xmat_train, Y_train, Xmat_val=None, Y_val=None, max_epochs=50, verbose=False):
        """
        Fit parameters of the neural network to given data using SGD.
        SGD ends after reaching a maximum number of epochs.

        Can optionally take in validation inputs as well to test generalization error.
        """
        
        # initialize all parameters randomly
        for p in self.parameters():
            p.data = random.uniform(-1, 1)
        
        n,d = Xmat_train.shape
        grad_vec = np.zeros(d)
        # iterate over epochs
        for e in range(max_epochs):
            
            shuffled_list = np.arange(0, n, 1).tolist()
            random.shuffle(shuffled_list)
            
            for i in shuffled_list:
                y = Y_train[i]
                yhat = self(Xmat_train[i])

                self._zero_grad()
                negative_loglikelihood(y, yhat).backward()

                for p in self.parameters():
                    p.data = p.data - self.learning_rate*(p.grad)

                

            # use the verbose flag with True when debugging your outputs
            if verbose:
                self.train_mode = False

                train_acc = accuracy(Y_train, self.predict(Xmat_train))
                
                if Xmat_val is not None:
                    val_acc = accuracy(Y_val, self.predict(Xmat_val))
                    print(f"Epoch {e}: Training accuracy {train_acc:.0f}%, Validation accuracy {val_acc:.0f}%")
                else:
                    print(f"Epoch {e}: Training accuracy {train_acc:.0f}%")

                self.train_mode = True

        # at the end of training set train mode to be False
        self.train_mode = False

    def predict(self, Xmat):
        """
        Predict method which returns a list of 0/1 labels for the given inputs
        """

        return [int(self(x).data > 0.5) for x in Xmat]


def chess_data():
    """
    Function to analyze chess data
    """
    data = pd.read_csv("games.csv")

    #Removing irrelevant columns (player and game id's don't have an impact on the model)
    data = data.drop(columns = ['white_id', 'black_id', 'id'])

    #Removing all players that don't have a rating
    data = data.drop(data[data['rated'] == False].index)

    #Games with less than 10 moves may be harmful to the model, and games with too many moves may add too many features, so we will remove games with < 10 or > 100 moves
    data = data.drop(data[data['turns'] < 10].index)
    data = data.drop(data[data['turns'] > 100].index)
    

    #splitting moves into their own variables
    splitMoves = data['moves'].str.split(expand = True)
    numCols = splitMoves.shape[1]
    splitMoves.columns = ['move ' + str(i) for i in range(1,numCols+1)]
    data = pd.concat([data, splitMoves], axis = 1)

    #Adding an average rating (average rating between black and white players), this will also be the prediction variables
    data['AverageRating'] = (data['white_rating'] + data['black_rating']) / 2
    
    #Xmat and Y values
    Xmat = data.drop(columns=['AverageRating', 'white_rating', 'black_rating']).to_numpy(dtype=np.float64)
    Y = data["AverageRating"].to_numpy(dtype=np.float64)

    #model training + validation
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.2, random_state=42)
    n, d = Xmat_train.shape
       
    #Standardizing data
    means = np.mean(Xmat_train, axis=0)
    stdevs = np.std(Xmat_train, axis=0)
    Xmat_train = (Xmat_train - means)/stdevs
    Xmat_val = (Xmat_val - means)/stdevs
    Xmat_test = (Xmat_test - means)/stdevs

    #Returning predictions on the test
    model = MLP(n_features=d, layer_sizes=[8, 4, 1], dropout_proba = 0.5)
    model.fit(Xmat_train, Y_train, Xmat_val = Xmat_val, Y_val = Y_val, max_epochs=50, verbose=False)


    return model, Xmat_test, Y_test

def main():
  
    #####################
    # Chess data
    #####################
    random.seed(42)
    model, X_test, Y_test = chess_data()

    # test final model
    test_acc = accuracy(Y_test, model.predict(X_test))
    print(f"Chess test accuracy {test_acc:.0f}%")

if __name__ == "__main__":
    main()
