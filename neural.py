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
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout) ]

    def __call__(self, x):
        return [ neuron(x) for neuron in self.neurons ]
    
    def parameters(self):
        return [ p for neuron in self.neurons for p in neuron.parameters() ]
    
class MLP:

    def __init__(self, nin, nouts):
        _layerSizes = [nin] + nouts
        self.layers = [Layer(_layerSizes[i], _layerSizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x if len(x) != 1 else x[0]
    
    def parameters(self):
        return [ p for layer in self.layers for p in layer.parameters() ]

n = MLP(3, [4, 4, 1])    

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0 -1.0, 1.0]

step = 0.001

for i in range(1000):
    ypreds = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for yout, ygt in zip (ypreds, ys))
    print(loss)
    loss.backward()

    for p in n.parameters():
        p.data -= step * p.grad
        p.grad = 0.0