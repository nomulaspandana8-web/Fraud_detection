import numpy as np

def preprocess_input(input_dict, scaler, columns):

    input_data = np.zeros((1, len(columns)))

    for key, value in input_dict.items():
        if key in columns:
            index = columns.index(key)
            input_data[0][index] = value

    input_scaled = scaler.transform(input_data)

    return input_scaled