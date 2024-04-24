import numpy as np
from graphviz import Digraph

class Value:

    def __init__(self, data, operator='', operands=(), label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self.operator = operator
        self.operands = set(operands)
        self.label = label or str(data)

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, '+', (self, other), f"{self.label}+{other.label}")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, '*', (self, other), f"{self.label}*{other.label}")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance (other, (int, float)), "only supporting int and float powers right now"
        out = Value(self.data ** other, '**', (self,), f"{self.label}**{other}")

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(np.exp(self.data), 'exp', (self,), f"e^({self.label})")

        def _backward():
            self.grad += out.grad * out.data 

        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(np.tanh(self.data), 'tanh', (self,), f"tanh({self.label})")

        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self**-1) * other
    
    def backward(self):

        _visited = set()
        _topo = []
        def build_topo(node):
            if node not in _visited:
                _visited.add(node)
                for operand in node.operands:
                    build_topo(operand)
                _topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(_topo):
            node._backward()
            
        
def graph(value):
    open = set()
    open.add(value)
    visited = set()
    graph = Digraph(format='svg', graph_attr={'rankdir' : 'LR'})

    while len(open):
        node = open.pop()
        if node not in visited:
            visited.add(node)
            uid = str(id(node))
            opuid = uid + node.operator
            graph.node(name=uid, label=f"{node.label} | data={node.data:.4f} | grad={node.grad:.4f}", shape='record')
            if node.operator:
                graph.node(name=opuid, label=f"{node.operator}")
                graph.edge(opuid, uid)
            for operand in node.operands:
                open.add(operand)
                graph.edge(str(id(operand)), opuid)

    return graph