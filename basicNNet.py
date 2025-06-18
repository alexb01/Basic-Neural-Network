import numpy as np
import random
import os
import csv

OUTPUT_DATA_CSV = "/Users/ab/Projects/Basic-Neural-Network/generated_output.csv"

# Network dimensions
input_dim = 1
hidden_layer_dim = 10
output_dim = 1

def generate_random_in_range():
    while True:
        r = random.random()

        if (0 < r <= 0.25) or (0.75 <= r < 1.0):
            return r

def gen_data():
    data_array = []

    for i in range(50001):
        x = generate_random_in_range()
        data_array.append((x,0 if x < 0.5 else 1))
    
    numpy_array_to_save = np.array(data_array)

    # delimiter=',' specifies CSV format
    # fmt='%f,%d' specifies the format for each column: float for the first, integer for the second
    np.savetxt(OUTPUT_DATA_CSV, numpy_array_to_save, delimiter=',', fmt='%f,%d')
    # If your second column could also be a float, you might use fmt='%.8f,%.0f' or just '%.8f,%.8f'
    # Or for mixed types, sometimes '%s' works, but '%f,%d' is explicit and better.

    return data_array


def train_network(training_data):
    # Loop through input datapoints:
    for i in range(0,len(training_data)):
        # Forward pass of input with L1 weights
        L1_array = training_data[i]['X_Value'] * W1 + bias1
        # Activation function for hidden layer
        a1 = sigmoid(L1_array)
        # Matrix multiply with W2 and add second bias
        a2 = a1 @ W2 + bias2

        Y_mse = 0.5 * (a2 - training_data[i]['Z_Label']) ** 2

def sigmoid(in_matrix):
    return 1 / (1 + np.exp(-(in_matrix)))

if __name__ == "__main__":
    print("Program start:\n")

    if not os.path.exists(OUTPUT_DATA_CSV):
        print(f"CSV data not found at {OUTPUT_DATA_CSV}\n")
        dataA = gen_data()
    else:
        # Open the existing file:
        print(f"CSV data found at {OUTPUT_DATA_CSV}, loading...\n")
        dataA = np.loadtxt(OUTPUT_DATA_CSV, delimiter=',', dtype={'names': ('X_Value', 'Z_Label'),
                                                                    'formats': ('f8', 'i4')})

    W1 = np.random.randn(input_dim, hidden_layer_dim) * np.sqrt(2.0 / (input_dim + hidden_layer_dim))
    bias1 = np.zeros((input_dim, hidden_layer_dim))
    W2 = np.random.randn(hidden_layer_dim, output_dim) * np.sqrt(2.0 / (hidden_layer_dim + output_dim))
    bias2 = np.zeros((1, output_dim))


