import numpy as np
import random
import os
import csv
import sys
from sklearn.datasets import make_moons
import ast
import argparse

class neural_net:
    def __init__(self,layer_dims):
        # [Input layer dimension [0], hidden layer dimension [1], output layer dimension [2]]
        self.layers = layer_dims
        print(f'Initializing network with layer dimensions {layer_dims}')
        self.num_layers = len(layer_dims)
        self.parameters = self._initialize_parameters()
        self.accuracy_cache = {}

    
    def _initialize_parameters(self):
        params = {}
        for l in range(1,self.num_layers):
            weight_key = f'W{l}'
            bias_key = f'b{l}'

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

            if np.any(np.isnan(A_prev)) or np.any(np.isinf(A_prev)):
                print(f"Layer {l}: NaN/Inf in A_prev")
                print(f"A_prev stats: min={np.min(A_prev)}, max={np.max(A_prev)}")
                
            if np.any(np.isnan(W)) or np.any(np.isinf(W)):
                print(f"Layer {l}: NaN/Inf in W{l}")
                print(f"W{l} stats: min={np.min(W)}, max={np.max(W)}")
                
            if np.any(np.isnan(b)) or np.any(np.isinf(b)):
                print(f"Layer {l}: NaN/Inf in b{l}")
                
            # Check for extreme values that might cause overflow
            if np.max(np.abs(A_prev)) > 1e10:
                print(f"Layer {l}: Extreme values in A_prev: max_abs = {np.max(np.abs(A_prev))}")
                
            if np.max(np.abs(W)) > 1e10:
                print(f"Layer {l}: Extreme values in W{l}: max_abs = {np.max(np.abs(W))}")

            Z = A_prev @ W + b
            
            # Check Z after computation
            if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
                print(f"Layer {l}: NaN/Inf in Z{l} after computation")
                print(f"Z{l} stats: min={np.min(Z)}, max={np.max(Z)}")

            if l == self.num_layers - 1:
                # Last layer
                A = self._sigmoid(Z)
                self.cache[f'activation{l}'] = 'sigmoid'
            else:
                # Each of the other layers
                A = self._relu(Z)
                self.cache[f'activation{l}'] = 'relu'

            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        return A


    def _checkshape(self, y_hat, y_true):
        if y_true.shape == (y_hat.shape[0],):  # Only 1D arrays like (200,)
            # 1D array with correct length - reshape to column vector
            y_true = y_true.reshape(-1, 1)
        elif y_true.shape[0] != y_hat.shape[0]:
            # Different number of samples
            print(f'self._backprop() Error: y_hat has {y_hat.shape[0]} samples, y_true has {y_true.shape[0]} samples. Exiting.\n')
            sys.exit()
        elif y_true.ndim > 1 and y_true.shape[1] != 1:
            # 2D but not a column vector
            print(f'self._backprop() Error: y_true should be a column vector, got shape {y_true.shape}. Exiting.\n')
            sys.exit()
        
        return y_true


    def _backprop(self, y_hat, y_true):
        m = y_hat.shape[0]  # batch size
        L = self._calcloss(y_hat, y_true)
        
        gradients = {
            'dLdZ': {},
            'dLdA': {},
            'dLdW': {}, 
            'dLdb': {}
        }

        y_true = self._checkshape(y_hat,y_true)

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
                if np.any(np.isnan(gradients['dLdZ'][l])) or np.any(np.isinf(gradients['dLdZ'][l])):
                    print(f"Layer {l}:")
                    print(f"dLdZ[{l}] stats: min={np.min(gradients['dLdZ'][l])}, max={np.max(gradients['dLdZ'][l])}")
                    print(f"W{l} stats: min={np.min(self.parameters[f'W{l}'])}, max={np.max(self.parameters[f'W{l}'])}")
                    print(f"dLdZ has NaN: {np.any(np.isnan(gradients['dLdZ'][l]))}")
                    print(f"dLdZ has inf: {np.any(np.isinf(gradients['dLdZ'][l]))}")
                
                gradients['dLdA'][l-1] = gradients['dLdZ'][l] @ self.parameters[f'W{l}'].T
                
                if np.any(np.isnan(gradients['dLdA'][l-1])) or np.any(np.isinf(gradients['dLdA'][l-1])):
                    print(f"After matmul - dLdA[{l-1}] has NaN: {np.any(np.isnan(gradients['dLdA'][l-1]))}")
                    print(f"After matmul - dLdA[{l-1}] has inf: {np.any(np.isinf(gradients['dLdA'][l-1]))}")

        return gradients

    
    def _update_params(self, gradients, learning_rate=0.05):
        for i in range(1,self.num_layers):
            self.parameters[f'W{i}'] -= learning_rate * gradients['dLdW'][i]
            self.parameters[f'b{i}'] -= learning_rate * gradients['dLdb'][i]


    def _calcloss(self, y_hat, y_true):
        # self.cache['MSE'] = np.mean(0.5 * (y_hat - y_true) ** 2)
        y_true = self._checkshape(y_hat,y_true)
        
        epsilon = 1e-15
        y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
        self.cache['BCE'] = -np.mean(y_true * np.log(y_hat_clipped) + (1 - y_true) * np.log(1 - y_hat_clipped))
        
        return self.cache['BCE'] # if self.cache['activation1'] == 'relu' else self.cache['MSE']


    def test_network(self, X, y):
        y_hat = self._forward_pass(X)
        y_true = self._checkshape(y_hat, y)

        # 3 to make it 3x wider than the accuracy percentage as a decimal.
        self.accuracy_cache['accuracy_threshold'] = 3 * self.accuracy_cache['middle'] / (self.accuracy_cache['near_zero'] + self.accuracy_cache['middle'] + self.accuracy_cache['near_one'])
        yhat_pushed = np.where(y_hat < self.accuracy_cache['accuracy_threshold'], 0, np.where(y_hat > (1 - self.accuracy_cache['accuracy_threshold']), 1, y_hat))

        self._checkaccuracy(yhat_pushed, y_true, self.accuracy_cache['accuracy_threshold'])


    def _checkaccuracy(self, y_hat, y_true, accuracy_threshold):
        accuracy = 100 * (np.count_nonzero(y_hat == y_true) / len(y_true))

        print(f'Accuracy for {len(y_true)} data points = {accuracy:.2f}% with accuracy threshold {accuracy_threshold}')


    def train_network(self, X, y, epochs=500, batch_size=32):
        num_samples = len(X)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        self.accuracy_cache['near_zero'] = 0
        self.accuracy_cache['middle'] = 0
        self.accuracy_cache['near_one'] = 0

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

                self._update_params(batch_grad, learning_rate=0.01)

                batch_loss = self._calcloss(y_hat, current_batch_y)
                total_epoch_loss += batch_loss

                self.accuracy_cache['near_zero'] += np.sum((y_hat >= 0) & (y_hat <= 0.1))
                self.accuracy_cache['middle'] += np.sum((y_hat > 0.1) & (y_hat < 0.9))
                self.accuracy_cache['near_one'] += np.sum((y_hat >= 0.9) & (y_hat <= 1))

            if epoch % 50 == 0:
                avg_loss = total_epoch_loss / num_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}, Batch Loss: {batch_loss:.6f}")
                print(f"Near 0: {100 * self.accuracy_cache['near_zero'] / (self.accuracy_cache['near_zero'] + self.accuracy_cache['middle'] + self.accuracy_cache['near_one']):.2f}%, Middle: {100 * self.accuracy_cache['middle'] / (self.accuracy_cache['near_zero'] + self.accuracy_cache['middle'] + self.accuracy_cache['near_one']):.2f}%, Near 1: {100 * self.accuracy_cache['near_one'] / (self.accuracy_cache['near_zero'] + self.accuracy_cache['middle'] + self.accuracy_cache['near_one']):.2f}%")


if __name__ == "__main__":
    print("Program start:")

    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="A neural network class.")
    parser.add_argument("--network_dims", type=str, default="[2,10,1]", help="Optional: Network layer dimensions as \"[2,10,1]\"")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Optional: The number of training epochs.")
    parser.add_argument("--num_train_samples", type=int, default=1000, help="Optional: The number of training samples.")
    parser.add_argument("--num_test_samples", type=int, default=200, help="Optional: The number of test samples.")
    parser.add_argument("--noise", type=float, default=0.15, help="Optional: Make_moons noise.")
    # parser.add_argument("l_rate", type=float, default=0.05, help="The learning rate for the network")
    args = parser.parse_args()

    # Convert string to list
    try:
        network_dims = ast.literal_eval(args.network_dims)
        num_epochs = args.num_epochs
        num_train_samples = args.num_train_samples
        num_test_samples = args.num_test_samples
        mm_noise = args.noise
        print(f'{network_dims=}\n{num_epochs=}\n{num_train_samples=}\n{num_test_samples=}\n{mm_noise=}\n')
    except:
        print("Invalid argument(s). Exiting.")
        sys.exit()

    net1 = neural_net(network_dims)
    moons_X_train, moons_y_train = make_moons(n_samples=num_train_samples, noise=mm_noise, random_state=1)

    net1.train_network(moons_X_train, moons_y_train, epochs=num_epochs)

    moons_X_test, moons_y_test = make_moons(n_samples=num_test_samples, noise=mm_noise, random_state=115)

    net1.test_network(moons_X_test, moons_y_test)