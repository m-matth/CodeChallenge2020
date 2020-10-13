import os
import pygame
import argparse
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from DQN2 import DQNAgent2
from DQN3 import DQNAgent3
from random import randint
from keras.utils import to_categorical
import random
import statistics


# x = 30   (*20px = 600px)
# y = 15   (*20px = 300px)

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/75
    params['learning_rate'] = 0.0005
 #   params['first_layer_size'] = 64   # neurons in the first layer
    params['first_layer_size'] = 50   # neurons in the first layer
 #   params['second_layer_size'] = 512   # neurons in the second layer
    params['second_layer_size'] = 300   # neurons in the second layer
 #    params['third_layer_size'] = 256    # neurons in the third layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 600
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/test_mouse_cat.hdf5' # 'weights/weights3.hdf5'
    params['load_weights'] = False
    params['train'] = True
    params['plot_score'] = True
    params['temp_save_path'] = 'weights_temp/'
    return params


class Game:
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('CodeChallenge')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.size_x = math.floor(game_width / 20)
        self.size_y = math.floor(game_height / 20)
        self.walls = Walls(self)
        self.player = Player(self)
        self.enemy = Enemy(self)
        self.enemy2 = Enemy(self)
        self.score = 0
        self.bullet = Bullet(self)

    def get_state_variables(self, x, y):
        radius = 3
        enemies = list(filter(None, [ self.enemy.coord, self.enemy2.coord ]))
        zippedPE = list(zip([self.player.coord]*len(enemies), enemies))

        has_enemy_on_left = True in list(map(lambda pe: pe[0][1] == pe[1][1] and abs(pe[0][0] - pe[1][0]) <= radius and pe[0][0] > pe[1][0], zippedPE))
        has_enemy_on_right = True in list(map(lambda pe: pe[0][1] == pe[1][1] and abs(pe[0][0] - pe[1][0]) <= radius and pe[0][0] < pe[1][0], zippedPE))
        has_enemy_on_top = True in list(map(lambda pe: pe[0][0] == pe[1][0] and abs(pe[0][1] - pe[1][1]) <= radius and pe[0][1] < pe[1][1], zippedPE))
        has_enemy_on_bottom = True in list(map(lambda pe: pe[0][0] == pe[1][0] and abs(pe[0][1] - pe[1][1]) <= radius and pe[0][1] < pe[1][1], zippedPE))        
        has_bullet = self.player.bullet_left
        has_bullet_cooldown = self.player.bullet_cooldown

        has_wall_on_left =  (x - 1, y) in self.walls.coords 
        has_wall_on_right = (x + 1, y) in self.walls.coords 
        has_wall_on_top = (x, y - 1) in self.walls.coords 
        has_wall_on_bottom = (x, y + 1) in self.walls.coords
        score = self.score
        map_ = self.get_state_pov_old(x, y)
        state = [
            has_enemy_on_left,
            has_enemy_on_right,
            has_enemy_on_top,
            has_enemy_on_bottom,
            has_bullet,
            has_bullet_cooldown,
            has_wall_on_left,
            has_wall_on_right,
            has_wall_on_top,
            has_wall_on_bottom,
            score
        ]
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
#        state_ = state + map_
#        return np.asarray(state_)
        return np.asarray(state)

    def get_state_pov_old(self, x, y):
        pov_radius = 3
        pov_line = pov_radius + 1 + pov_radius
        state = [0] * (pov_line**2) # player centered (7*7 = 49)
        
        self_idx = (pov_line * pov_radius) + pov_radius
        state[self_idx] = 2 # '*' # you are here
        for (direction, (bx, by)) in self.bullet.coords:
            if abs(x - bx) <= pov_radius and abs(y - by) <= pov_radius:
                idx = self_idx + (bx - x)
                idx += (by - y) * (pov_line)
                state[idx] = 3 # 'b'
        for (wx, wy) in self.walls.coords:
            if abs(x - wx) <= pov_radius and abs(y - wy) <= pov_radius:
                idx = self_idx + (wx - x)
                idx += (wy - y) * (pov_line)
                state[idx] = 4 # '#'
        if self.enemy.coord is not None:
            ex, ey = self.enemy.coord
            if abs(x - ex) <= pov_radius and abs(y - ey) <= pov_radius:
                idx = self_idx + (ex - x)
                idx += (ey - y) * (pov_line)
                state[idx] = 5 # 'e'
        if self.enemy2.coord is not None:
            ex, ey = self.enemy2.coord
            if abs(x - ex) <= pov_radius and abs(y - ey) <= pov_radius:
                idx = self_idx + (ex - x)
                idx += (ey - y) * (pov_line)
                state[idx] = 6 # 'e'

        if (x - pov_radius - 1) < 0:
           for y_ in range(pov_line):
               for x_ in range(pov_radius - x):
                   state[x_ + (y_ * pov_line)] = 7 # '~'
        if (x + pov_radius + 1) >= self.size_x:
           for y_ in range(pov_line):
               for x_ in range(x + 1, x + pov_radius + 1):
#               for x_ in range(x + (self.size_x - x) + 1, x + pov_radius + 3):
#                   print('x =' + str(x_) + ' ' + 'y = ' + str(y_) + ' end : ' + str(x + pov_radius + 1) + ' size_x ' + str(self.size_x)) 
                       #                   print(x_ + (y_ * pov_line))
                   if (x_ >= self.size_x):
                       idx = self_idx + (abs(x_ - x))
                       idx += (abs(y_ - y)) * (pov_line)
                       if idx >= 0 and idx < (pov_line**2):
                           state[idx] = 8 # '~'                
#                       state[x_ + (y_ * pov_line)] = '@'
#               for x_ in range((pov_radius + x) - self.size_x):
#                   state[(x + x_) + (y_ * pov_line)] = '@'

        if (y - pov_radius - 1) < 0:
           for x_ in range(pov_line):
               for y_ in range(pov_radius - y):
                   state[x_ + (y_ * pov_line)] = 9 # '~'
        if (y + pov_radius + 1) >= self.size_y:
           for x_ in range(pov_line):
               # for y_ in range(y + (self.size_y - y) + 1, y + pov_radius + 3):
               for y_ in range(y + 1, y + pov_radius + 1):
                   if (y_ >= self.size_y):
                       idx = self_idx + (abs(x_ - x))
                       idx += (abs(y_ - y)) * (pov_line)
                       if idx >= 0 and idx < pov_line**2:
                           state[idx] = 10 # '~'
                       # state[x_ + (y_ * pov_line)] = '?'
#               for y_ in range((pov_radius + y) - self.size_y):
#                   state[x_ + ((y + y_) * pov_line)] = '?'

#        for p in range(pov_line):
#            for x in range(pov_line):
#                print(state[x + (p*pov_line)], end = '')
#            print('\n', end = '')
#        print('\n', end = '')

#        return np.asarray(state)
        return state
#
#        pov = 3
#
# '*' : player
# 'e' : enemy
# 'b' : bullet
# '#' : wall
# '.' : outside map
#
# 0  # # # # # # # 6
# 7  #   #     # # 13
# 14 #   # #     # 20
# 21 #     *     # 27
# 28 #       #   # 34
# 35 #   #     e # 41
# 42 # # # # # # # 48
#
#

#
# pov outside border of map 
#
#    . . . . . . .
#    . . . . . . .
#    . . # # # # #
#    . . # *   #  
#    . . #     b #
#    . . #       #
#    . . # e      
#
#

class Bullet(object):
    def __init__(self, game):
        self.image_up = pygame.image.load('img/fire_up.png')
        self.image_down = pygame.image.load('img/fire_down.png')
        self.image_right = pygame.image.load('img/fire_right.png')
        self.image_left = pygame.image.load('img/fire_left.png')
        self.coords = []

    def new(self, x, y, direction):
        if direction == 40:
            self.coords.append((direction, (x + 1, y)))
        elif direction == 41:
            self.coords.append((direction, (x - 1, y)))
        elif direction == 42:
            self.coords.append((direction, (x, y - 1)))
        elif direction == 43:
            self.coords.append((direction, (x, y + 1)))

    def hit(self, x, y):
        for (direction, (bx, by)) in self.coords:
            if x == bx and y == by:
                return True
        return False
        
    def display_bullets(self, game):
        updated_coord = []
        if game.crash == False:
            for (direction, (x, y)) in self.coords:
#                game.gameDisplay.blit(self.image, (x * 20, y * 20))
                if (x,y) not in game.walls.coords:
                    if direction == 40:
                        game.gameDisplay.blit(self.image_right, (x * 20, y * 20))
                        updated_coord.append((direction, (x + 1, y)))
                    elif direction == 41:
                        game.gameDisplay.blit(self.image_left, (x * 20, y * 20))
                        updated_coord.append((direction, (x - 1, y)))
                    elif direction == 42:
                        game.gameDisplay.blit(self.image_up, (x * 20, y * 20))
                        updated_coord.append((direction, (x, y - 1)))
                    elif direction == 43:
                        game.gameDisplay.blit(self.image_down, (x * 20, y * 20))
                        updated_coord.append((direction, (x, y + 1)))

            self.coords = updated_coord
            update_screen()
        else:
            pygame.time.wait(300)

class Walls(object):
    def __init__(self, game):
        # 30% of space other than border are wall
        nbOfWalls = math.floor(((game.size_x - 2) * (game.size_y - 2)) * 0.18)
        self.coords = self.generateBorder(game)
        max_attempt = 20
        for w in range(nbOfWalls):
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
            attempt = 0
            while (x_candidate, y_candidate) in self.coords and attempt < max_attempt:
                attempt += 1
                x_candidate = randint(1, game.size_x - 2)
                y_candidate = randint(1, game.size_y - 2)
            if attempt < max_attempt:
                self.coords.append((x_candidate, y_candidate))
        self.image = pygame.image.load('img/wall_.png')

    def generateBorder(self, game):
        border = []
        for x in range(game.size_x):
            border.append((x, 0))
            border.append((x, game.size_y - 1))
        for y in range(game.size_y):
            border.append((0, y))
            border.append((game.size_x - 1, y))
        return border

    def display_walls(self, game):
        if game.crash == False:
            for c in self.coords:
                game.gameDisplay.blit(self.image, (c[0] * 20, c[1] * 20))
            update_screen()
        else:
            pygame.time.wait(300)
        
class Player(object):
    def __init__(self, game):
        self.enemy = 1
        self.killed = False
        self.invalid_mvt = False
        self.image = pygame.image.load('img/player_.png')
#        self.x_change = 1 # 20
#        self.y_change = 1
        x_candidate = randint(1, game.size_x - 2)
        y_candidate = randint(1, game.size_y - 2)
        max_attempt = 20
        self.bullet_left = 10
        self.bullet_cooldown = 0
        attempt = 0
        while (x_candidate, y_candidate) in game.walls.coords and attempt < max_attempt:
            attempt += 1
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
        if attempt < max_attempt:
            self.coord = (x_candidate, y_candidate)

    def do_move(self, x, y, game, enemy, agent, state_old):
        new_coord = (0, 0)
        attempt = 0
        max_attempt = 3
        while new_coord in game.walls.coords and attempt < max_attempt:
            attempt += 1
            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                move = to_categorical(randint(0, 7), num_classes=8)
            else:
                # predict action based on the old state
                # prediction = agent.model.predict(state_old.reshape((1, 49)))
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                move = to_categorical(np.argmax(prediction[0]), num_classes=8)

            self.x_change = 0
            self.y_change = 0
            if np.array_equal(move, [1, 0, 0, 0, 0, 0, 0, 0]):  # right - going horizontal
                new_coord = (x + 1, y)
                self.x_change, self.y_change = [1, 0]
            elif np.array_equal(move, [0, 1, 0, 0, 0, 0, 0, 0]):  # right - going vertical
                new_coord = (x, y + 1)
                self.x_change, self.y_change = [0, 1]
            elif np.array_equal(move, [0, 0, 1, 0, 0, 0, 0, 0]):  # left - going horizontal
                new_coord = (x, y -1)
                self.x_change, self.y_change = [0, -1]
            elif np.array_equal(move, [0, 0, 0, 1, 0, 0, 0, 0]):  # left - going vertical
                new_coord = (x - 1 , y)
                self.x_change, self.y_change = [-1, 0]
            elif self.bullet_cooldown == 0 and np.array_equal(move, [0, 0, 0, 0, 1, 0, 0, 0]):  # fire up
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x,y-1) not in game.walls.coords:
                    game.bullet.new(x, y, 40)
                    self.bullet_left -= 1
                    self.bullet_cooldown = 6
                else:
                    self.invalid_mvt = True
            elif self.bullet_cooldown == 0 and np.array_equal(move, [0, 0, 0, 0, 0, 1, 0, 0]):  # fire down
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x,y+1) not in game.walls.coords:
                    game.bullet.new(x, y, 41)
                    self.bullet_left -= 1
                    self.bullet_cooldown = 6
                else:
                    self.invalid_mvt = True
            elif self.bullet_cooldown == 0 and np.array_equal(move, [0, 0, 0, 0, 0, 0, 1, 0]):  # fire left
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x-1,y) not in game.walls.coords:
                    game.bullet.new(x, y, 42)
                    self.bullet_left -= 1
                    self.bullet_cooldown = 6
                else:
                    self.invalid_mvt = True
            elif self.bullet_cooldown == 0 and np.array_equal(move, [0, 0, 0, 0, 0, 0, 0, 1]):  # fire right
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x+1,y) not in game.walls.coords:
                    game.bullet.new(x, y, 43)
                    self.bullet_left -= 1
                    self.bullet_cooldown = 6
                else:
                    self.invalid_mvt = True
            elif self.bullet_cooldown > 0:
                continue

        if self.bullet_cooldown > 0:
            self.bullet_cooldown -= 1
                    
        if new_coord not in game.walls.coords:
            self.coord = new_coord

        if game.bullet.hit(self.coord[0], self.coord[1]) is True:
            self.killed = True
            print('killed !')
            
#        else:
#            self.invalid_mvt = True
        return move
            
    def do_move2(self, move, x, y, game, enemy, agent):
        move_array = [self.x_change, self.y_change]

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        new_coord = (x + self.x_change, y + self.y_change)
        if new_coord in game.walls.coords:
            self.invalid_mvt = True
        else:
            self.coord = new_coord
        
    def display_player(self, x, y, enemy, game):
        if game.crash == False:
            game.gameDisplay.blit(self.image, (x * 20, y * 20))
            update_screen()
        else:
            pygame.time.wait(300)

class Opponent(object):
    def __init__(self, game):
        self.image = pygame.image.load('img/opponent_.png')
        self.x_change = 1
        self.y_change = 0
        self.killed = False
        max_attempt = 20
        attempt = 0
        x_candidate = randint(1, game.size_x - 2)
        y_candidate = randint(1, game.size_y - 2)
        while (x_candidate, y_candidate) in game.walls.coords and attempt < max_attempt:
            attempt += 1
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
        if attempt < max_attempt:
            self.coord = (x_candidate, y_candidate)

    def set_reward(self, final_move):
        # train short memory base on the new action and state
        agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
        # store the new data into a long term memory
        agent.remember(state_old, final_move, reward, state_new, game.crash)

    def do_move(self, game, agent):       
        # perform random actions based on agent.epsilon, or choose the action
        if random.uniform(0, 1) < agent.epsilon:
            move = to_categorical(randint(0, 3), num_classes=4)
        else:
            # predict action based on the old state
            prediction = agent.model.predict(state_old.reshape((1, 11)))
            move = to_categorical(np.argmax(prediction[0]), num_classes=4)
      
        move_array = [self.x_change, self.y_change]

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]

        self.x_change, self.y_change = move_array
        new_coord = (self.coord[0] + self.x_change, self.coord[1] + self.y_change)
        if new_coord not in game.walls.coords:
            self.coord = new_coord

    def display_opponent(self, x, y, game):
        game.gameDisplay.blit(self.image, (x * 20, y * 20))
        update_screen()
    
class Enemy(object):
    def __init__(self, game):
        self.image = pygame.image.load('img/enemy_.png')
        self.x_change = 1
        self.y_change = 0
        self.killed = 0
        self.spawn(game)
        self.bullet_left = 10

    def spawn(self, game):
        max_attempt = 20
        attempt = 0
        x_candidate = randint(1, game.size_x - 2)
        y_candidate = randint(1, game.size_y - 2)
        while (x_candidate, y_candidate) in game.walls.coords and attempt < max_attempt:
            attempt += 1
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
        self.coord = (x_candidate, y_candidate)

    def set_reward(self, player, game, agent, final_move, state_new, state_old):
        reward = 0
        if self.coord is None:
            reward = -10
        else:
            if self.coord == player.coord:
                reward = -10
                if self.coord[0] >= player.coord[0] - 1 and self.coord[0] <= player.coord[0] + 1:
                    reward = -1
                if self.coord[1] >= player.coord[1] - 1 and self.coord[1] <= player.coord[1] + 1:
                    reward -= 1
                if abs(self.coord[0] - player.coord[0]) > 5:
                    reward = 5
                if abs(self.coord[1] - player.coord[1]) > 5:
                    reward += 5

        # train short memory base on the new action and state
        agent.enemy_train_short_memory(state_old, final_move, reward, state_new, game.crash)
        # store the new data into a long term memory
        agent.enemy_remember(state_old, final_move, reward, state_new, game.crash)

            
    def do_move(self, game, enemy, agent):
        new_coord = (0, 0)
        attempt = 0
        max_attempt = 20

        if self.coord is None:
            self.spawn(game)

        (x, y) = self.coord

        while new_coord in game.walls.coords and attempt < max_attempt:
            attempt += 1
            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                move = to_categorical(randint(0, 7), num_classes=8)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 12)))
                move = to_categorical(np.argmax(prediction[0]), num_classes=8)

#            move_array = [self.x_change, self.y_change]
            self.x_change = 0
            self.y_change = 0

            if np.array_equal(move, [1, 0, 0, 0, 0, 0, 0, 0]):  # right - going horizontal
                new_coord = (x + 1, y)
                self.x_change, self.y_change = [1, 0]
            elif np.array_equal(move, [0, 1, 0, 0, 0, 0, 0, 0]):  # right - going vertical
                new_coord = (x, y + 1)
                self.x_change, self.y_change = [0, 1]
            elif np.array_equal(move, [0, 0, 1, 0, 0, 0, 0, 0]):  # left - going horizontal
                new_coord = (x, y -1)
                self.x_change, self.y_change = [0, -1]
            elif np.array_equal(move, [0, 0, 0, 1, 0, 0, 0, 0]):  # left - going vertical
                new_coord = (x - 1 , y)
                self.x_change, self.y_change = [-1, 0]
            elif np.array_equal(move, [0, 0, 0, 0, 1, 0, 0, 0]):  # fire up
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x,y-1) not in game.walls.coords:
                    game.bullet.new(x, y, 40)
                    self.bullet_left -= 1
                else:
                    self.invalid_mvt = True
            elif np.array_equal(move, [0, 0, 0, 0, 0, 1, 0, 0]):  # fire down
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x,y+1) not in game.walls.coords:
                    game.bullet.new(x, y, 41)
                    self.bullet_left -= 1
                else:
                    self.invalid_mvt = True
            elif np.array_equal(move, [0, 0, 0, 0, 0, 0, 1, 0]):  # fire left
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x-1,y) not in game.walls.coords:
                    game.bullet.new(x, y, 42)
                    self.bullet_left -= 1
                else:
                    self.invalid_mvt = True
            elif np.array_equal(move, [0, 0, 0, 0, 0, 0, 0, 1]):  # fire right
                new_coord = (x , y)
                if self.bullet_left >= 1 and (x+1,y) not in game.walls.coords:
                    game.bullet.new(x, y, 43)
                    self.bullet_left -= 1
                else:
                    self.invalid_mvt = True

#            if np.array_equal(move, [1, 0, 0]):
#                move_array = self.x_change, self.y_change
#            elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
#                move_array = [0, self.x_change]
#            elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
#                move_array = [-self.y_change, 0]
#            elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
#                move_array = [0, -self.x_change]
#            elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
#                move_array = [self.y_change, 0]
#            self.x_change, self.y_change = move_array
#            new_coord = (x + self.x_change, y + self.y_change)

        if new_coord not in game.walls.coords:
            self.coord = new_coord
        else:
            print('(enemy) in wall')
        if game.bullet.hit(self.coord[0], self.coord[1]) is True:
            self.killed += 1
            self.coord = None # will respawn next turn
            print('Hit !')

        return move

    def display_enemy(self, game):
        if self.coord is not None:
            (x, y) = self.coord
            game.gameDisplay.blit(self.image, (x * 20, y * 20))
            update_screen()

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    #    game.gameDisplay.blit(text_score, (45, 440))
    #    game.gameDisplay.blit(text_score_number, (120, 440))
    #    game.gameDisplay.blit(text_highest, (190, 440))
    #    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (0, 0))
    game.walls.display_walls(game)


def display(player, enemy, enemy2, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.coord[0], player.coord[1], player.enemy, game)
    enemy.display_enemy(game)
    enemy2.display_enemy(game)
    game.bullet.display_bullets(game)

def update_screen():
    pygame.display.update()


def initialize_game(player, game, enemy, enemy2, agent, enemy_agent, batch_size):
    # state_init1 = agent.get_state(game, player, enemy)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    state_init1 = game.get_state_variables(player.coord[0], player.coord[1])
    action = [1, 0, 0]
    player.do_move(player.coord[0], player.coord[1], game, enemy, agent, state_init1)
    # state_init2 = agent.get_state(game, player, enemy)
    state_init2 = game.get_state_variables(player.coord[0], player.coord[1])
    reward1 = agent.set_reward(player, enemy, enemy2)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)
    enemy_agent.enemy_replay_new(agent.enemy_memory, batch_size)


def plot_seaborn(array_counter, array_score,train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    if train==False:
        fit_reg = False
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.ylim(0,65)
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


# def test(display_option, speed, player_weights, enemy_weights, params):
#    params['load_weights'] = True
#    params['train'] = False
#    score, mean, stdev = run(display_option, speed, player_weights, enemy_weights, params)
#    return score, mean, stdev


def run(display_option, speed, player_weights, enemy_weights, params):
    pygame.init()
    agent = DQNAgent3(params)
    enemy_agent = DQNAgent2(params)
    opponent_agent = DQNAgent(params)
    weights_filepath = params['weights_path']

    if player_weights != False:
        agent.model.load_weights(player_weights)
        print(f'player weights {player_weights} loaded')
    if enemy_weights != False:
        enemy_agent.enemy_model.load_weights(enemy_weights)
        print(f'enemy weights {enemy_weights} loaded')

    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        #game = Game(600, 300)
        #        game = Game(300, 150)
        game = Game(300, 300)
        player1 = game.player
        enemy1 = game.enemy
        enemy2 = game.enemy2
#        opponent1 = game.opponent

        # Perform first move
        initialize_game(player1, game, enemy1, enemy2, agent, enemy_agent, params['batch_size'])
        if display_option:
            display(player1, enemy1, enemy2, game, record)
        mvt = 0
        
        while not game.crash:
            mvt = mvt + 1
            if not params['train']:
                agent.epsilon = 0.00
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            # state_old = agent.get_state(game, player1, enemy1)
            enemy_state_old = enemy_agent.enemy_get_state(game, player1, enemy1)

            state_old = game.get_state_variables(player1.coord[0], player1.coord[1])

            # perform random actions based on agent.epsilon, or choose the action
#            if random.uniform(0, 1) < agent.epsilon:
#                final_move = to_categorical(randint(0, 2), num_classes=3)
#            else:
#                # predict action based on the old state
#                prediction = agent.model.predict(state_old.reshape((1, 11)))
#                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # perform new move and get new state
            enemy_final_move = enemy1.do_move(game, enemy1, enemy_agent)
            enemy2_final_move = enemy2.do_move(game, enemy1, enemy_agent)
#            enemy2.do_move(final_move, enemy2.coord[0], enemy2.coord[1], game, enemy2, agent)
#            opp_move = opponent1.do_move(game, opponent_agent)
            final_move = player1.do_move(player1.coord[0], player1.coord[1], game, enemy1, agent, state_old)
#            state_new = agent.get_state(game, player1, enemy1)
            state_new = game.get_state_variables(player1.coord[0], player1.coord[1])
            enemy_state_new = enemy_agent.enemy_get_state(game, player1, enemy1)

#            opponent_agent.set_reward(opp_move)
            # set reward for the new state
            reward = agent.set_reward(player1, enemy1, enemy2)
            enemy_reward = enemy1.set_reward(player1, game, enemy_agent, enemy_final_move, enemy_state_new, enemy_state_old)
            # print(reward)
#            if reward == -20:
#                game.crash = True
            if enemy1.coord == player1.coord or enemy2.coord == player1.coord:
                print("crushed..")
                game.crash = True
            if mvt > 100:
                game.crash = True

            if player1.killed == True:
                game.crash = True

#            if player1.invalid_mvt == True:
#                game.crash = True

            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)
            game.score += reward
            record = get_record(game.score, record)
            if display_option:
                display(player1, enemy1, enemy2, game, record)
                pygame.time.wait(speed)
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
        if counter_games % 100 == 0:
            print('saving..')
            agent.model.save_weights(params['temp_save_path'] + str(counter_games) + '_player.hdf5')
            enemy_agent.enemy_model.save_weights(params['temp_save_path'] + str(counter_games) + '_enemy.hdf5')

    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        agent.model.save_weights(params['weights_path'])
        enemy_agent.enemy_model.save_weights('enemy_v0.hdf5')
        # total_score, mean, stdev = test(display_option, speed, player_weights, enemy_weights, params)
    # if params['plot_score']:
    #    plot_seaborn(counter_plot, score_plot, params['train'])
    # print('Total score: {}   Mean: {}   Std dev:   {}'.format(total_score, mean, stdev))
    # return total_score, mean, stdev


if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", type=bool, default=False)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--enemy", type=str, default=False)
    parser.add_argument("--player", type=str, default=False)
    args = parser.parse_args()
    params['bayesian_optimization'] = False    # Use bayesOpt.py for Bayesian Optimization
    run(args.display, args.speed, args.player, args.enemy, params)
