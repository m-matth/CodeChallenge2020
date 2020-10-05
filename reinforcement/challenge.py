import os
import pygame
import argparse
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
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
    params['first_layer_size'] = 50   # neurons in the first layer
    params['second_layer_size'] = 300   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 150           
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights3.hdf5'
    params['load_weights'] = False
    params['train'] = True
    params['plot_score'] = True
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


class Walls(object):
    def __init__(self, game):
        # 30% of space other than border are wall
        nbOfWalls = math.floor(((game.size_x - 2) * (game.size_y - 2)) * 0.30)
        self.coords = self.generateBorder(game)
        for w in range(nbOfWalls):
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
            while (x_candidate, y_candidate) in self.coords:
                x_candidate = randint(1, game.size_x - 2)
                y_candidate = randint(1, game.size_y - 2)
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
        self.eaten = False
        self.invalid_mvt = False
        self.image = pygame.image.load('img/player_.png')
        self.x_change = 1 # 20
        self.y_change = 0
        x_candidate = randint(1, game.size_x - 2)
        y_candidate = randint(1, game.size_y - 2)
        while (x_candidate, y_candidate) in game.walls.coords:
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
        self.coord = (x_candidate, y_candidate)

    def do_move(self, move, x, y, game, enemy, agent):
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


class Enemy(object):
    def __init__(self, game):
        self.image = pygame.image.load('img/enemy_.png')
        self.x_change = 1
        self.y_change = 0

        x_candidate = randint(1, game.size_x - 2)
        y_candidate = randint(1, game.size_y - 2)
        while (x_candidate, y_candidate) in game.walls.coords:
            x_candidate = randint(1, game.size_x - 2)
            y_candidate = randint(1, game.size_y - 2)
        self.coord = (x_candidate, y_candidate)


    def do_move(self, move, x, y, game, enemy, agent):
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

        new_coord = (x + self.x_change, y + self.y_change)
        if new_coord not in game.walls.coords:
            self.coord = new_coord

    def display_enemy(self, x, y, game):
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
    enemy.display_enemy(enemy.coord[0], enemy.coord[1], game)
    enemy2.display_enemy(enemy2.coord[0], enemy2.coord[1], game)

def update_screen():
    pygame.display.update()


def initialize_game(player, game, enemy, agent, batch_size):
    state_init1 = agent.get_state(game, player, enemy)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.coord[0], player.coord[1], game, enemy, agent)
    state_init2 = agent.get_state(game, player, enemy)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


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


def test(display_option, speed, params):
    params['load_weights'] = True
    params['train'] = False
    score, mean, stdev = run(display_option, speed, params)
    return score, mean, stdev


def run(display_option, speed, params):
    pygame.init()
    agent = DQNAgent(params)
    weights_filepath = params['weights_path']
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
        game = Game(600, 300)
        player1 = game.player
        enemy1 = game.enemy
        enemy2 = game.enemy2

        # Perform first move
        initialize_game(player1, game, enemy1, agent, params['batch_size'])
        if display_option:
            display(player1, enemy1, enemy2, game, record)

        while not game.crash:
            if not params['train']:
                agent.epsilon = 0.00
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1, enemy1)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # perform new move and get new state
            enemy1.do_move(final_move, enemy1.coord[0], enemy1.coord[1], game, enemy1, agent)
            enemy2.do_move(final_move, enemy2.coord[0], enemy2.coord[1], game, enemy2, agent)
            player1.do_move(final_move, player1.coord[0], player1.coord[1], game, enemy1, agent)
            state_new = agent.get_state(game, player1, enemy1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            # print(reward)
            if reward == -20:
                game.crash = True

            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

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
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        agent.model.save_weights(params['weights_path'])
        total_score, mean, stdev = test(display_option, speed, params)
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    print('Total score: {}   Mean: {}   Std dev:   {}'.format(total_score, mean, stdev))
    return total_score, mean, stdev


if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", type=bool, default=False)
    parser.add_argument("--speed", type=int, default=50)
    args = parser.parse_args()
    params['bayesian_optimization'] = False    # Use bayesOpt.py for Bayesian Optimization
    run(args.display, args.speed, params)
