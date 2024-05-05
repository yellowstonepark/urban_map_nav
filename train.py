import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from collections import deque
import random
import os

# Helper functions to manage grid state
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def is_corner(i, j):
    return (i % 3 != 1) and (j % 3 != 1)

def is_sidewalk(i, j):
    return (i % 3 == 1 and j % 3 != 1) or (i % 3 != 1 and j % 3 == 1)

def valid_actions(state, grid):
    actions = []
    x, y = state
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if is_corner(nx, ny) or is_sidewalk(nx, ny):
                actions.append((nx, ny))
    return actions

def calculate_reward(current_state, next_state, grid):
    # Extract the attributes of the next state cell
    attributes = grid[next_state[0], next_state[1]]
    reward = 0

    if is_sidewalk(next_state[0], next_state[1]):
        # Sidewalk rewards and penalties
        width, density, construction = attributes
        reward += (width - 1) * 1.0  # Width reward: Higher width gives higher reward
        reward -= density * 0.5      # Density penalty: Higher density reduces reward
        if construction == 1:
            reward -= 2.0            # Construction penalty: Construction present

    elif is_corner(next_state[0], next_state[1]):
        # Corner penalties based on difficulty of crossing
        signal_type, difficulty_lr, difficulty_ud = attributes
        reward -= (difficulty_lr + difficulty_ud) * 0.5  # Difficulty penalty: Harder crossing reduces reward

    elif grid[next_state[0], next_state[1], 0] == 2:  # Assuming '2' signifies a block
        # Blocks are heavily penalized or treated as impassable
        return -float('inf')  # Infinite penalty for choosing a block (if impassable)

    return reward


def state_to_input(state, grid):
    return np.ravel(grid[max(0, state[0]-1):state[0]+2, max(0, state[1]-1):state[1]+2, :])

# DQN Model
def create_dqn_model(input_shape, action_space):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


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
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    data = load_data('map_data/map_data.pkl')
    num_episodes = 1000
    agent = DQNAgent(state_size=4, action_size=4)  # Adjust
    
    for e in range(num_episodes):
        for grid in data:
            state = (0, 0)  # Starting position, adjust as necessary
            state_input = state_to_input(state, grid)
            for time in range(500):  # Adjust the maximum steps
                actions = valid_actions(state, grid)
                # There will always be at least two actions
                action = agent.act(state_input)

                if action >= len(actions):
                    #print("Model did not pick a valid action, picking a random one")
                    action = np.random.randint(len(actions))
                #else:
                #    print("Valid action taken")

                next_state = valid_actions(state, grid)[action]
                reward = calculate_reward(state, next_state, grid)

                done = next_state == (grid.shape[0]-1, grid.shape[1]-1)  # terminal state
                next_state_input = state_to_input(next_state, grid)

                agent.remember(state_input, action, reward, next_state_input, done)
                state = next_state
                state_input = next_state_input

                if done:
                    print("episode: {}/{}, score: {}".format(e, num_episodes, time))
                    break

            #agent.replay(32)  # Batch size
