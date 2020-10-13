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
        self.enemy_memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network(49)
        self.enemy_model = self.network(11)

    def network2(self, dim):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(dim,10,10,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
        layer4 = layers.Flatten()(layer3)
        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(8, activation="linear")(layer5)
        return keras.Model(inputs=inputs, outputs=action)

    def network(self, dim):
        model = Sequential()
        #        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=dim))
        model.add(Dense(activation="relu", input_dim=dim, units=self.first_layer))
        model.add(Dense(activation="relu", units=self.second_layer))
        model.add(Dense(activation="relu", units=self.third_layer))
        model.add(Dense(activation="softmax", units=8))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def network3(self, dim):
        model = Sequential()
        #        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=dim))
        #        model.add(Dense(activation="relu", input_dim=dim, units=50))
        model.add(Dense(activation="relu", input_shape=((dim,5)), units=50))
        #        model.add(Dense(output_dim=self.second_layer, activation='relu'))
        model.add(Dense(activation="relu", units=300))
        #        model.add(Dense(output_dim=self.third_layer, activation='relu'))
        model.add(Dense(activation="relu", units=50))
        #        model.add(Dense(output_dim=8, activation='softmax'))
        model.add(Dense(activation="softmax", units=8))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
#        state = [
#            (player.x_change == 1 and player.y_change == 0 and  ((list(map(add, player.coord, [1, 0])) in [player.coord]) or
#            player.coord[0] + 1 >= (game.game_width - 1))) or (player.x_change == -1 and player.y_change == 0 and ((list(map(add, player.coord, [-1, 0])) in [player.coord]) or
#            player.coord[0] - 1 < 1)) or (player.x_change == 0 and player.y_change == -1 and ((list(map(add, player.coord, [0, -1])) in [player.coord]) or
#            player.coord[-1] - 1 < 1)) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add, player.coord, [0, 1])) in [player.coord]) or
#            player.coord[-1] + 1 >= (game.game_height-1))),  # danger straight

#            (player.x_change == 0 and player.y_change == -1 and ((list(map(add,player.coord,[1, 0])) in [player.coord]) or
#            player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,
#            [-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == -1 and player.y_change == 0 and ((list(map(
#            add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
#            (list(map(add,player.coord,[0,1])) in [player.coord]) or player.coord[-1] + 1 >= (game.game_height-1))),  # danger right

#             (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,[1,0])) in [player.coord]) or
#             player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == -1 and ((list(map(
#            add, player.coord,[-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
#            (list(map(add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (
#            player.x_change == -1 and player.y_change == 0 and ((list(map(add,player.coord,[0,1])) in [player.coord]) or
#            player.coord[-1] + 1 >= (game.game_height-1))), #danger left
#            ]

#     def get_state(self, game, player, enemy):
#         state = [
#             (player.x_change == 1 and player.y_change == 0 and  ((list(map(add, player.coord, [1, 0])) in [player.coord]) or
#             player.coord[0] + 1 >= (game.game_width - 1))) or (player.x_change == -1 and player.y_change == 0 and ((list(map(add, player.coord, [-1, 0])) in [player.coord]) or
#             player.coord[0] - 1 < 1)) or (player.x_change == 0 and player.y_change == -1 and ((list(map(add, player.coord, [0, -1])) in [player.coord]) or
#             player.coord[-1] - 1 < 1)) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add, player.coord, [0, 1])) in [player.coord]) or
#             player.coord[-1] + 1 >= (game.game_height-1))),  # danger straight

#             (player.x_change == 0 and player.y_change == -1 and ((list(map(add,player.coord,[1, 0])) in [player.coord]) or
#             player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,
#             [-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == -1 and player.y_change == 0 and ((list(map(
#             add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
#             (list(map(add,player.coord,[0,1])) in [player.coord]) or player.coord[-1] + 1 >= (game.game_height-1))),  # danger right

#              (player.x_change == 0 and player.y_change == 1 and ((list(map(add,player.coord,[1,0])) in [player.coord]) or
#              player.coord[0] + 1 > (game.game_width-1))) or (player.x_change == 0 and player.y_change == -1 and ((list(map(
#             add, player.coord,[-1,0])) in [player.coord]) or player.coord[0] - 1 < 1)) or (player.x_change == 1 and player.y_change == 0 and (
#             (list(map(add,player.coord,[0,-1])) in [player.coord]) or player.coord[-1] - 1 < 1)) or (
#             player.x_change == -1 and player.y_change == 0 and ((list(map(add,player.coord,[0,1])) in [player.coord]) or
#             player.coord[-1] + 1 >= (game.game_height-1))), #danger left
            
#             player.x_change == -1,  # move left
#             player.x_change == 1,  # move right
#             player.y_change == -1,  # move up
#             player.y_change == 1,  # move down
#             enemy.coord[0] < player.coord[0], # enemy left
#             enemy.coord[0] > player.coord[0],  # enemy right
# #            enemy.x < player.x,  # enemy left
# #            enemy.x > player.x,  # enemy right
#             enemy.coord[1] < player.coord[1],  # enemy up
#             enemy.coord[1] > player.coord[1]  # enemy down
# #            enemy.y < player.y,  # enemy up
# #            enemy.y > player.y  # enemy down
#             ]

#         for i in range(len(state)):
#             if state[i]:
#                 state[i]=1
#             else:
#                 state[i]=0

#         return np.asarray(state)

    def set_reward(self, player, enemy, enemy2):
        self.reward = -1

        if enemy.coord is not None and abs(player.coord[0] - enemy.coord[0]) < 5 and abs(player.coord[1] - enemy.coord[1]) < 5:
            self.reward = 1

        if enemy2.coord is not None and abs(player.coord[0] - enemy2.coord[0]) < 5 and abs(player.coord[1] - enemy2.coord[1]) < 5:
            self.reward = 1

        if enemy.coord is None or enemy2.coord is None:
            self.reward = 35

        if enemy.coord == player.coord or enemy2.coord == player.coord:
            self.reward = -20

        if player.killed:
            self.reward = -20

#        if enemy.coord is not None and abs(player.coord[0] - enemy.coord[0]) < 4 and player.coord[1] == enemy.coord[1]:
#            self.reward = 2
#        if enemy.coord is not None and abs(player.coord[1] - enemy.coord[1]) < 4 and player.coord[0] == enemy.coord[0]:
#            self.reward = 2

#        if enemy.coord is not None and abs(player.coord[0] - enemy.coord[0]) < 5 and abs(player.coord[1] - enemy.coord[1]) < 5:
#            self.reward = 1

#        if player.bullet_left <= 0:
#            self.reward = -2

#        if killed:
#            self.reward = -10
#            return self.reward
#        if player.invalid_mvt:
#            self.reward -= 50
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
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 49)))[0])
        target_f = self.model.predict(state.reshape((1, 49)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 49)), target_f, epochs=1, verbose=0)
