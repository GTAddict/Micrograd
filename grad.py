import math
import numpy as np

class Value:

    def __init__(self, data, operator, operands):
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
        
    

