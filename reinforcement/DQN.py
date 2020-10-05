from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections

class DQNAgent(object):
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
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=11))
        model.add(Dense(output_dim=self.second_layer, activation='relu'))
        model.add(Dense(output_dim=self.third_layer, activation='relu'))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
    def get_state(self, game, player, enemy):
        state = [
            (player.x_change == 1 and player.y_change == 0 and  ((list(map(add, player.coord, [1, 0])) in [player.coord]) or
            player.coord[0] + 1 >= (game.game_width - 1))) or (player.x_change == -1 and player.y_change == 0 and ((list(map(add, player.coord, [-1, 0])) in [player.coord]) or
            player.coord[0] - 1 < 1)) or (player.x_change == 0 and player.y_change == -1 and ((list(map(add, player.coord, [0, -1])) in [player.coord]) or
            player.coord[-1] - 1 < 1)) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add, player.coord, [0, 1])) in [player.coord]) or
            player.coord[-1] + 1 >= (game.game_height-1))),  # danger straight

            (player.x_change == 0 and player.y_change == -1 and ((list(map(add,player.coord,[1, 0])) in [player.coord]) or
            player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,
            [-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == -1 and player.y_change == 0 and ((list(map(
            add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
            (list(map(add,player.coord,[0,1])) in [player.coord]) or player.coord[-1] + 1 >= (game.game_height-1))),  # danger right

             (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,[1,0])) in [player.coord]) or
             player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == -1 and ((list(map(
            add, player.coord,[-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
            (list(map(add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (
            player.x_change == -1 and player.y_change == 0 and ((list(map(add,player.coord,[0,1])) in [player.coord]) or
            player.coord[-1] + 1 >= (game.game_height-1))), #danger left
            
            player.x_change == -1,  # move left
            player.x_change == 1,  # move right
            player.y_change == -1,  # move up
            player.y_change == 1,  # move down
            enemy.coord[0] < player.coord[0], # enemy left
            enemy.coord[0] > player.coord[0],  # enemy right
#            enemy.x < player.x,  # enemy left
#            enemy.x > player.x,  # enemy right
            enemy.coord[1] < player.coord[1],  # enemy up
            enemy.coord[1] > player.coord[1]  # enemy down
#            enemy.y < player.y,  # enemy up
#            enemy.y > player.y  # enemy down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, killed):
        self.reward = 0
        if killed:
            self.reward = -10
            return self.reward
        if player.invalid_mvt:
            self.reward = -20
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
