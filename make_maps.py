import numpy as np
import os

def generate_map_data(num_maps, map_size=(3, 3)):
    data = np.random.randint(0, 4, size=(num_maps, map_size[0], map_size[1], 3))
    return data

def save_maps_to_disk(data, batch_size=10000, folder='map_data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data))
        batch_data = data[start:end]
        np.save(os.path.join(folder, f'map_batch_{i}.npy'), batch_data)

num_maps = 1000000
batch_size = 10000 

for batch_start in range(0, num_maps, batch_size):
    batch_end = min(batch_start + batch_size, num_maps)
    map_data = generate_map_data(batch_end - batch_start)
    save_maps_to_disk(map_data, batch_size=batch_size)

print("done")
