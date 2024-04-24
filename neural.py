import numpy as np
import random
from grad import Value, graph

class Neuron:

    def __init__(self, num):
        self.w = [ Value(random.uniform(-1.0, 1.0)) for _ in range(num)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x):
        y = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        act = y.tanh()
        return act
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout) ]

    def __call__(self, x):
        return [ neuron(x) for neuron in self.neurons ]
    
class MLP:

    def __init__(self, nin, nouts):
        _layerSizes = [nin] + nouts
        self.layers = [Layer(_layerSizes[i], _layerSizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x if len(x) != 1 else x[0] 
    
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
o = n(x)
o.backward()
graph(o).view()
