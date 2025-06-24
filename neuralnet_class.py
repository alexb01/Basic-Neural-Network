import numpy as np
import random
import os
import csv
import sys

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

        A = X
        self.cache['A0'] = X

        for l in range(1,self.num_layers):
            A_prev = A

            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = A_prev @ W + b

            if l == self.num_layers - 1:
                A = self._sigmoid(Z)
                self.cache[f'activation{l}'] = 'sigmoid'
            else:
                A = self._relu(Z)
                self.cache[f'activation{l}'] = 'relu'

            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        return A


    def _backprop(self, y_hat, y_true):
        L = self._calcloss(y_hat,y_true)
        dLdZ = {}
        dLdA = {}
        dWdZ = {}
        dLdb = {}

        for l in range(self.num_layers-1,0,-1):
            if l == self.num_layers - 1 and 'BCE' in self.cache:
                dLdZ[f'Z{l}'] = (y_hat - y_true) * self._d_sigmoid(self.cache['Z{l}'])
                dLdA[f'A{l}'] = 
            elif l < self.num_layers - 1:
                dLdZ[f'Z{l}'] = 
            else:
                # If this runs, either the loss type was changed from BCE to something else without accounting for it,
                # or _calcloss() wasn't called somehow.
                print("Only code for binary cross entropy has been written, but self.cache doesn't have stored BCE values. Exiting program.")
                sys.exit()
                
            dWdZ[f'W{l}'] = self.cache[f'A{l}'] @ dLdZ[f'{l}']


    def _calcloss(self, y_hat, y_true):
        self.cache['BCE'] = np.mean(-(y_true * np.log(y_hat) + (1 - y_true) * np.log(1-y_hat)))
        # self.cache['MSE'] = np.mean(0.5 * (y_hat - y_true) ** 2)
        return self.cache['BCE']
    
