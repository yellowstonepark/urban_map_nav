import pickle
import numpy as np

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    file_path = 'map_data/map_data.pkl'  
    maps = load_data_from_pickle(file_path)

    second_map = maps[99456]

    specific_element = second_map[1][6][2]

    print("Specific element is:", specific_element)
