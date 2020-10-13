from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import pandas as pd
from operator import add
import collections

class DQNAgent2(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.enemy_memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.enemy_model = self.network(12)

    def network(self, dim):
        model = Sequential()
        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=dim))
        model.add(Dense(output_dim=self.second_layer, activation='relu'))
        model.add(Dense(output_dim=self.third_layer, activation='relu'))
        model.add(Dense(output_dim=8, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
    def enemy_get_state(self, game, player, enemy):
        state = [
            (enemy.x_change == 1 and enemy.y_change == 0),
            (enemy.x_change == 0 and enemy.y_change == -1),
            (enemy.x_change == 0 and enemy.y_change == 1),
            enemy.x_change == -1,
            enemy.x_change == 1,
            enemy.y_change == -1,
            enemy.y_change == 1,
            (enemy.coord is not None and enemy.coord[0] > player.coord[0]) or 0, # player left
            (enemy.coord is not None and enemy.coord[0] < player.coord[0]) or 0, # player right
            (enemy.coord is not None and enemy.coord[1] > player.coord[1]) or 0, # player up
            (enemy.coord is not None and enemy.coord[1] < player.coord[1]) or 0,  # player down
            enemy.coord is None
            ]
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        return np.asarray(state)

    def set_reward(self, player, enemy):
        self.reward = enemy.killed * 25

        if enemy.coord == player.coord:
            self.reward = -25

        if enemy.coord is not None and abs(player.coord[0] - enemy.coord[0]) < 4 and player.coord[1] == enemy.coord[1]:
            self.reward = 2
        if enemy.coord is not None and abs(player.coord[1] - enemy.coord[1]) < 4 and player.coord[0] == enemy.coord[0]:
            self.reward = 2

#        if enemy.coord is not None and abs(player.coord[0] - enemy.coord[0]) < 5 and abs(player.coord[1] - enemy.coord[1]) < 5:
#            self.reward = 1

#        if player.bullet_left <= 0:
#            self.reward = -2

#        if killed:
#            self.reward = -10
#            return self.reward
        if player.invalid_mvt:
            self.reward = -50
            return self.reward
#        if player.eaten:
#            self.reward = 10
        return self.reward

    def enemy_remember(self, state, action, reward, next_state, done):
        self.enemy_memory.append((state, action, reward, next_state, done))

    def enemy_replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.enemy_model.predict(np.array([next_state]))[0])
            target_f = self.enemy_model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.enemy_model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            
    def enemy_train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.enemy_model.predict(next_state.reshape((1, 12)))[0])
        target_f = self.enemy_model.predict(state.reshape((1, 12)))
        target_f[0][np.argmax(action)] = target
        self.enemy_model.fit(state.reshape((1, 12)), target_f, epochs=1, verbose=0)

