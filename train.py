import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from collections import deque
import random
import os
from multiprocessing import Pool
import time

# Use metal GPU for macOS

# Helper functions to manage grid state
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def is_corner(grid, i, j):
    return ((i % 3 != 1) and (j % 3 != 1)) and i < grid.shape[0] and j < grid.shape[1]

def is_sidewalk(grid, i, j):
    return ((i % 3 == 1 and j % 3 != 1) or (i % 3 != 1 and j % 3 == 1)) and i < grid.shape[0] and j < grid.shape[1]

def valid_actions(state, grid):
    actions = []
    x, y = state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if is_corner(grid, nx, ny) or is_sidewalk(grid, nx, ny):
                actions.append((nx, ny))
    return actions

def calculate_reward(current_state, next_state, target, grid):
    # Extract the attributes of the next state cell
    attributes = grid[next_state[0], next_state[1]]
    reward = 0

    # Calculate Manhattan distance to target from the current state and the next state
    current_dist = abs(current_state[0] - target[0]) + abs(current_state[1] - target[1])
    next_dist = abs(next_state[0] - target[0]) + abs(next_state[1] - target[1])

    # Reward improvement in distance to the target
    if next_dist < current_dist:
        reward += 1.0  # Reward for moving closer to the target
    elif next_dist > current_dist:
        reward -= 1.5  # Penalty for moving away from the target

    if is_sidewalk(grid, next_state[0], next_state[1]):
        # Sidewalk rewards and penalties
        width, density, construction = attributes
        reward += (width - 1) * 1.0  # Width reward: Higher width gives higher reward
        reward -= density * 0.5      # Density penalty: Higher density reduces reward
        if construction == 1:
            reward -= 2.0            # Construction penalty: Construction present

    elif is_corner(grid, next_state[0], next_state[1]):
        # Corner penalties based on difficulty of crossing
        signal_type, difficulty_lr, difficulty_ud = attributes
        reward -= (difficulty_lr + difficulty_ud) * 0.5  # Difficulty penalty: Harder crossing reduces reward

    elif grid[next_state[0], next_state[1], 0] == 2:  # Assuming '2' signifies a block
        # Blocks are heavily penalized or treated as impassable
        return -float('inf')  # Infinite penalty for choosing a block (if impassable)

    return reward

def state_to_input(state, target, grid):
    max_x, max_y = 42, 57  # Maximum grid sizes calculated from x = 15*3, y = 20*3
    padded_grid = np.full((max_x, max_y, 3), [-1, -1, -1], dtype=np.float32)  # Using -1 to fill
    
    # Position the original grid within the padded grid
    original_shape = grid.shape
    padded_grid[:original_shape[0], :original_shape[1], :] = grid
    
    # Include current and target positions as additional channels
    current_channel = np.zeros((max_x, max_y, 1))
    target_channel = np.zeros((max_x, max_y, 1))
    current_channel[state[0], state[1], 0] = 1  # Mark current position
    target_channel[target[0], target[1], 0] = 1  # Mark target position

    # Stack these additional channels with the grid
    input_grid = np.concatenate((padded_grid, current_channel, target_channel), axis=-1)
    
    return input_grid

# DQN Model
def create_dqn_model(input_shape, action_space):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Save model
def save_model(model, file_name='dqn_model.h5'):
    model.save(file_name)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_dqn_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print("Not enough memory to replay.")
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Reset memory after replay (empty deque)
        self.memory.clear()

def test_agent(model, test_maps, num_tests):
    directions = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

    for i in range(num_tests):
        grid = test_maps[i]  # Select a test map
        state = (np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1]))
        while grid[state[0], state[1], 0] == 2:  # Ensure the start is not a block
            state = (np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1]))

        target = (grid.shape[0]-1, grid.shape[1]-1)  # Set a target position
        print(f"Test {i+1}: Start at {state}, Target at {target}")

        steps = 0
        max_steps = 1000  # Set a limit to prevent infinite loops in problematic scenarios
        while state != target and steps < max_steps:
            state_input = state_to_input(state, target, grid)
            action = np.argmax(model.predict(state_input))
            print(f"Step {steps+1}: {directions[action]}")

            # Apply the action
            if action == 0 and state[0] > 0:  # Up
                state = (state[0] - 1, state[1])
            elif action == 1 and state[0] < grid.shape[0] - 1:  # Down
                state = (state[0] + 1, state[1])
            elif action == 2 and state[1] > 0:  # Left
                state = (state[0], state[1] - 1)
            elif action == 3 and state[1] < grid.shape[1] - 1:  # Right
                state = (state[0], state[1] + 1)

            steps += 1

        print(f"Reached target in {steps} steps" if state == target else "Failed to reach target within step limit")

        if state == target:
            print("Success: Reached the target.")
        else:
            print("Failure: Did not reach the target.")


def process_grid(data):
    grid, state, target, agent = data
    transitions = []  # To hold all transitions (state, action, reward, next_state, done)
    
    state_input = np.expand_dims(state_to_input(state, target, grid), axis=0)
    for time in range(500):  # Limit of 500 moves
        actions = valid_actions(state, grid)
        action = agent.act(state_input)  # Decision based on the current state input

        if action >= len(actions):
            action = np.random.randint(len(actions))

        next_state = actions[action]
        next_state_input = state_to_input(next_state, target, grid)
        next_state_input = np.expand_dims(next_state_input, axis=0)

        reward = calculate_reward(state, next_state, target, grid)
        done = next_state == target

        # Store the transition for this move
        transitions.append((state_input, action, reward, next_state_input, done, next_state))
        
        # Update the current state and input for the next iteration
        state = next_state
        state_input = next_state_input

        if done:
            break

    return transitions

def main():
    data = load_data('map_data/map_data.pkl')
    num_episodes = 300
    agent = DQNAgent(state_size=(42, 57, 5), action_size=4)

    for e in range(num_episodes):
        print(f"Episode {e+1}/{num_episodes}")
        start_time = time.time()
        process_data = [(grid, (0, 0), (np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])), agent) for grid in random.sample(data, 8)]
        
        with Pool(processes=2) as pool:
            results = pool.map(process_grid, process_data)
        
        # Process results from all grids, and update agent
        for grid_results in results:
            for state_input, action, reward, next_state_input, done, _ in grid_results:
                agent.remember(state_input, action, reward, next_state_input, done)

        agent.replay(128)  # Update the model based on the collected experiences
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # estimated time remaining in minutes rounded to 2 decimal places
        print(f"Estimated time remaining: {(num_episodes - e - 1) * (time.time() - start_time) / 60:.2f} minutes")


    save_model(agent.model, 'dqn_model.h5')

if __name__ == '__main__':
    main()

