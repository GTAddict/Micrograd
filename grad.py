import math
import numpy as np
from graphviz import Digraph

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
                graph.node(name=uid, label=f"data={node.data}", shape='record')
                if node.operator:
                    graph.node(name=opuid, label=f"{node.operator}")
                    graph.edge(opuid, uid)
                for operand in node.operands:
                    open.add(operand)
                    graph.edge(str(id(operand)), opuid)

        return graph

class Value:

    def __init__(self, data, operator='', operands=()):
        self.data = data
        self.operator = operator
        self.operands = set(operands)

    def __repr__(self):
        return f"Value(data={self.data}, operator={self.operator}, operands={self.operands})"
    
    def __add__(self, other):
        return Value(self.data + other.data, '+', (self, other))
    
    def __sub__(self, other):
        return Value(self.data - other.data, '-', (self, other))
    
    def __mul__(self, other):
        return Value(self.data * other.data, '*', (self, other))
    
    def __truediv__(self, other):
        return Value(self.data / other.data, '/', (self, other))
        
       

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print (d)
graph = graph(d)
graph.view()
                    

                    

