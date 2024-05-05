import numpy as np
import os
import pickle
from multiprocessing import Pool, cpu_count

def generate_sidewalk(arr, i, j):
    # [width (0-2), density (0-3), construction (0-1)]
    arr[i, j] = np.array([np.random.randint(0, 3), np.random.randint(0, 4), np.random.randint(0, 2)])

def generate_corner(arr, i, j):
    # [signal type (0-1), difficulty LR (0-2), difficulty UD (0-2)]
    arr[i, j] = np.array([np.random.randint(0, 2), np.random.randint(0, 3), np.random.randint(0, 3)])

    #if within bounds, check the sidewalk above and to the left and mirror the difficulty and signal type
    if i > 0:
        arr[i-1, j, 0] = arr[i, j, 0]
        arr[i-1, j, 1] = arr[i, j, 1]
        arr[i-1, j, 2] = arr[i, j, 2]
    if j > 0:
        arr[i, j-1, 0] = arr[i, j, 0]
        arr[i, j-1, 1] = arr[i, j, 1]
        arr[i, j-1, 2] = arr[i, j, 2]

def generate_block(arr, i, j):
    # Calculate average busyness from sidewalks
    
    avg_busyness = round((arr[i+1, j, 1] + arr[i, j+1, 1] + arr[i-1, j, 1] + arr[i, j-1, 1]) / 4 + .5)
    
    # [length (700-1000 ft), width (700-1000 ft), avg busyness]
    arr[i, j] = np.array([np.random.randint(700, 1001), np.random.randint(700, 1001), avg_busyness])

def generate_map(dummy):
    # Generates a grid of blocks
    x = np.random.randint(7,15) * 3
    y = np.random.randint(5,20) * 3
    arr = np.zeros((x,y,3), dtype=int)
    
    for i in range(x):
        for j in range(y):
                if (i % 3 == 1): 
                    # The repeating pattern is: s, b, s, s, b, s, s, b, s, s, b,
                    if (j % 3 != 1):
                        generate_sidewalk(arr, i, j)
                    else:
                        generate_block(arr, i, j)
                else:  # Odd rows
                    # The repeating pattern is: c, s, c, c, s, c, c, s, c, c, s,
                    if (j % 3 != 1):
                        generate_corner(arr, i, j)
                    else:
                        generate_sidewalk(arr, i, j)

    # regenerate every block, nothing else just the block
    for i in range(x):
        for j in range(y):
            if (i % 3 == 1 and j % 3 == 1):
                generate_block(arr, i, j)

    return arr

def save_to_disk(data, folder='map_data'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # pickle version
    with open(os.path.join(folder, 'map_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    # readable version
    with open(os.path.join(folder, 'map_data.txt'), 'w') as f:
        for i, map in enumerate(data):
            f.write(f'Map {i}\n')
            for row in map:
                for block in row:
                    f.write(f'{block}\n')
                f.write('\n')

# Generate a million maps
def generate_maps(num_maps):
    # Pool of workers, utilizing as many cores as available
    with Pool(processes=cpu_count()) as pool:
        # generate num_maps maps using generate_map function
        result = pool.map(generate_map, [None] * num_maps, chunksize=100)
    return result

if __name__ == '__main__':
    # Generate maps
    map_data = generate_maps(102111)

    print("Map data generated. Saving to disk...")

    # Save the generated map data to disk
    save_to_disk(map_data)

    print("Map data generated and saved. Check the 'map_data' folder.")

