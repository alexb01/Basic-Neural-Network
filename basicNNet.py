import numpy as np
import random
import os
import csv

OUTPUT_DATA_CSV = "/Users/ab/Projects/Basic-Neural-Network/generated_output.csv"

# Network dimensions
input_dim = 1
hidden_layer_dim = 10
output_dim = 1

W1 = np.random.randn(input_dim, hidden_layer_dim) * np.sqrt(2.0 / (input_dim + hidden_layer_dim))
bias1 = np.zeros((input_dim, hidden_layer_dim))
W2 = np.random.randn(hidden_layer_dim, output_dim) * np.sqrt(2.0 / (hidden_layer_dim + output_dim))
bias2 = np.zeros((1, output_dim))


def generate_random_in_range():
    while True:
        r = random.random()

        if (0 < r <= 0.25) or (0.75 <= r < 1.0):
            return r


def gen_test_data(num):
    data_array = []
    for i in range(num):
        x = generate_random_in_range()
        data_array.append((x,0 if x < 0.5 else 1))

    return np.array(data_array)

def gen_data(range_val):
    data_array = []

    for i in range(range_val):
        x = generate_random_in_range()
        data_array.append((x,0 if x < 0.5 else 1))
    
    numpy_array_to_save = np.array(data_array)

    # delimiter=',' specifies CSV format
    # fmt='%f,%d' specifies the format for each column: float for the first, integer for the second
    np.savetxt(OUTPUT_DATA_CSV, numpy_array_to_save, delimiter=',', fmt='%f,%d')
    # If your second column could also be a float, you might use fmt='%.8f,%.0f' or just '%.8f,%.8f'
    # Or for mixed types, sometimes '%s' works, but '%f,%d' is explicit and better.

    return data_array


def train_network(training_data, W1, bias1, W2, bias2, learning_rate=0.01):
    
    grad_W1 = np.zeros_like(W1)
    grad_bias1 = np.zeros_like(bias1)
    grad_W2 = np.zeros_like(W2)
    grad_bias2 = np.zeros_like(bias2)

    num_samples = len(training_data)
    
    # Loop through input datapoints:
    for i in range(0,len(training_data)):
        # Forward pass of input with L1 weights
        Z1 = np.array([[training_data[i]['X_Value']]]) @ W1 + bias1
        # Activation function for hidden layer
        A1 = sigmoid(Z1)
        # Matrix multiply with W2 and add second bias
        Z2 = A1 @ W2 + bias2

        A2 = sigmoid(Z2)

        L = 0.5 * (np.array([[training_data[i]['Z_Label']]]) - A2) ** 2
        dL_dZ2 = -(np.array([[training_data[i]['Z_Label']]]) - A2) * d_sigmoid(Z2)

        dLdW2 = A1.T @ dL_dZ2
        dLdb2 = dL_dZ2
        dLdW1 = training_data[i]['X_Value'] @ (dL_dZ2 @ W2 * d_sigmoid(Z1))
        dLdb1 = dL_dZ2 @ W2 * d_sigmoid(Z1)

        grad_W1 += dLdW1
        grad_bias1 += dLdb1
        grad_W2 += dLdW2
        grad_bias2 += dLdb2

    W1 -= learning_rate * (grad_W1 / num_samples)
    bias1 -= learning_rate * (grad_bias1 / num_samples)
    W2 -= learning_rate * (grad_W2 / num_samples)
    bias2 -= learning_rate * (grad_bias2 / num_samples)

    return W1, bias1, W2, bias2


def run_network(input_data, W1, bias1, W2, bias2):

    num_samples = len(input_data)
    
    results = []

    # Loop through input datapoints:
    for i in range(0,num_samples):
        # Forward pass of input with L1 weights
        Z1 = np.array([[input_data]]) @ W1 + bias1
        # Activation function for hidden layer
        A1 = sigmoid(Z1)
        # Matrix multiply with W2 and add second bias
        Z2 = A1 @ W2 + bias2

        A2 = sigmoid(Z2)

        print(f"Predicted output for run {i}: {A2}\n")
        results += A2

    return results


def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


if __name__ == "__main__":
    print("Program start:\n")

    W1 = np.random.randn(input_dim, hidden_layer_dim) * np.sqrt(2.0 / (input_dim + hidden_layer_dim))
    bias1 = np.zeros((input_dim, hidden_layer_dim))
    W2 = np.random.randn(hidden_layer_dim, output_dim) * np.sqrt(2.0 / (hidden_layer_dim + output_dim))
    bias2 = np.zeros((1, output_dim))

    if not os.path.exists(OUTPUT_DATA_CSV):
        print(f"CSV data not found at {OUTPUT_DATA_CSV}\n")
        dataA = gen_data(50001)
    else:
        # Open the existing file:
        print(f"CSV data found at {OUTPUT_DATA_CSV}, loading...\n")
        dataA = np.loadtxt(OUTPUT_DATA_CSV, delimiter=',', dtype={'names': ('X_Value', 'Z_Label'),
                                                                    'formats': ('f8', 'i4')})

    trained_W1, trained_bias1, trained_W2, trained_bias2 = train_network(dataA, W1, bias1, W2, bias2)

    # Generate some test data:
    test_array = gen_test_data(15)
    res1 = run_network(test_array, trained_W1, trained_bias1, trained_W2, trained_bias2)

