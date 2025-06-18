import numpy as np
import random
import os

OUTPUT_DATA_CSV = "/Users/ab/Projects/Basic-Neural-Network/generated_output.csv"

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

    np.save(OUTPUT_DATA_CSV,data_array)

    return data_array



if __name__ == "__main__":

    print("Program start:\n")

    if not os.path.exists(OUTPUT_DATA_CSV):
        print(f"CSV data not found at {OUTPUT_DATA_CSV}\n")
        print("Training/test data file does not exist yet, generating...\n")
        dataA = gen_data()
    else:
        dataA = []