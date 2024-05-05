import pickle

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

import numpy as np

def is_corner(i, j):
    return (i % 3 != 1) and (j % 3 != 1)

def is_sidewalk(i, j):
    return ((i % 3 == 1) and (j % 3 != 1)) or ((i % 3 != 1) and (j % 3 == 1))

def valid_actions(state, grid_shape):
    moves = []
    x, y = state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
            if is_corner(nx, ny) or (is_sidewalk(x, y) and is_corner(nx, ny)):
                moves.append((nx, ny))
    return moves

def q_learning_train(maps, num_maps=30, episodes=100, learning_rate=0.8, gamma=0.95):
    q_table = dict()  # Initialize Q-table

    for map_index in range(min(num_maps, len(maps))):
        grid = maps[map_index]
        for episode in range(episodes):
            state = (0, np.random.randint(grid.shape[1]))  # Start at random corner on the first row
            if not is_corner(state[0], state[1]):
                continue
            
            for step in range(10 * grid.shape[0] * grid.shape[1]):
                actions = valid_actions(state, grid.shape)
                if not actions:
                    break
                next_state = np.random.choice(len(actions))
                next_state = actions[next_state]

                reward = -1  # Penalty for each move
                old_value = q_table.get((state, next_state), 0)
                future_optimal_value = max(q_table.get((next_state, next_action), 0) for next_action in valid_actions(next_state, grid.shape))
                
                # Q-learning formula
                q_table[(state, next_state)] = old_value + learning_rate * (reward + gamma * future_optimal_value - old_value)
                
                state = next_state  # Move to the next state

                if state[0] == grid.shape[0] - 1:  # If reaches the last row
                    break

    return q_table

def save_model(q_table, file_name='q_table.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(q_table, f)

if __name__ == '__main__':
    maps = load_data_from_pickle('map_data/map_data.pkl')
    q_table = q_learning_train(maps)
    save_model(q_table)
