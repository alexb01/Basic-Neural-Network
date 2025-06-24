import numpy as np
import random
import os
import csv

class neural_net:
    def __init__(self,layer_dims):
        # [Input layer dimension [0], hidden layer dimension [1], output layer dimension [2]]
        self.layers = layer_dims
        self.num_layers = len(layer_dims)
        self.parameters = self._initialize_parameters()

    
    def _initialize_parameters(self):
        params = {}
        for l in range(1,self.num_layers):
            weight_key = f'W{l}'
            bias_key = f'bias{l}'

            n_in = self.layers[l-1]
            n_out = self.layers[l]

            params[weight_key] = np.random.randn(n_in,n_out) * np.sqrt(2.0 / (n_in))
            params[bias_key] = np.zeros((1,n_out))

        return params


    def _relu(self,x):
        return np.maximum(0,x)

    
    def _d_relu(self,x):
        return np.where(x > 0, 1, 0)


    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-(x)))


    def _d_sigmoid(self,x):
        return self._sigmoid(x) * (1-self._sigmoid(x))
    

    def _forward_pass(self, X):
        self.cache = {}

        A_prev = X
        self.cache['A0'] = X

        for l in range(1,self.num_layers):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = A_prev @ W + b

            if l == self.num_layers - 1:
                A = self._sigmoid(Z)
            else:
                A = self._relu(Z)

            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

            A_prev = A

        return self.cache['A3']


    def _backprop(self, X, Y):

    
