import numpy as np
import random
import os
import csv
import sys
from sklearn.datasets import make_moons

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
            bias_key = f'b{l}'

            n_in = self.layers[l-1]
            n_out = self.layers[l]

            params[weight_key] = np.random.randn(n_in,n_out) * np.sqrt(1.0 / (n_in))
            params[bias_key] = np.zeros((1,n_out))

        return params


    def _relu(self,x):
        return np.maximum(0,x)

    
    def _d_relu(self,x):
        return np.where(x > 0, 1, 0)


    def _sigmoid(self,x):
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))


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
        m = y_hat.shape[0]  # batch size
        L = self._calcloss(y_hat, y_true)
        
        gradients = {
            'dLdZ': {},
            'dLdA': {},
            'dLdW': {}, 
            'dLdb': {}
        }

        # Start from output layer and work backwards
        for l in range(self.num_layers-1, 0, -1):
            if l == self.num_layers - 1:
                # Output layer - for sigmoid + BCE loss
                gradients['dLdZ'][l] = (y_hat - y_true) / m
            else:
                # Hidden layers
                if self.cache[f'activation{l}'] == 'relu':
                    gradients['dLdZ'][l] = gradients['dLdA'][l] * self._d_relu(self.cache[f'Z{l}'])
                else:
                    gradients['dLdZ'][l] = gradients['dLdA'][l] * self._d_sigmoid(self.cache[f'Z{l}'])

            # Compute weight and bias gradients
            gradients['dLdW'][l] = self.cache[f'A{l-1}'].T @ gradients['dLdZ'][l]
            gradients['dLdb'][l] = np.sum(gradients['dLdZ'][l], axis=0, keepdims=True)
            
            # Compute gradient w.r.t. previous layer's activations (except for input layer)
            if l > 1:
                gradients['dLdA'][l-1] = gradients['dLdZ'][l] @ self.parameters[f'W{l}'].T

        return gradients

    
    def _update_params(self, gradients, learning_rate=0.05):
        for i in range(1,self.num_layers):
            self.parameters[f'W{i}'] -= learning_rate * gradients['dLdW'][i]
            self.parameters[f'b{i}'] -= learning_rate * gradients['dLdb'][i]


    def _calcloss(self, y_hat, y_true):
        epsilon = 1e-15
        y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)

        self.cache['BCE'] = -np.mean(y_true * np.log(y_hat_clipped) + (1 - y_true) * np.log(1 - y_hat_clipped))
        # self.cache['MSE'] = np.mean(0.5 * (y_hat - y_true) ** 2)
        return self.cache['BCE']
    

    def train_network(self, X, y, epochs=500, batch_size=32):
        num_samples = len(X)
        print(f"Num_samples: {num_samples}\n")
        num_batches = (num_samples + batch_size - 1) // batch_size

        leftover_datapoints = num_samples % batch_size

        for epoch in range(epochs):
            total_epoch_loss = 0
        
            shuffled_indices = np.random.permutation(num_samples)
            shuffled_X = X[shuffled_indices]
            shuffled_y = y[shuffled_indices]

            for i in range(num_batches):
                # i represents each batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples) # Ensure we don't go out of bounds for the last batch
                
                current_batch_X = shuffled_X[start_idx:end_idx]
                current_batch_y = shuffled_y[start_idx:end_idx]
                current_batch_size = len(current_batch_X)

                y_hat = self._forward_pass(current_batch_X)

                batch_grad = self._backprop(y_hat, current_batch_y)

                self._update_params(batch_grad)

                batch_loss = self._calcloss(y_hat, current_batch_y)
                total_epoch_loss += batch_loss
            
            if epoch % 50 == 0:
                avg_loss = total_epoch_loss / num_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

                
net1 = neural_net([2,10,10,10,1])
moons1 = make_moons(n_samples=200, noise=0.15, random_state=42)