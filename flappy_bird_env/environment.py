import time
import pygame
import gym
import gym.spaces
import enum

import numpy as np

from .bird import Bird
from .pipe import Pipe

# AI PARAMETERS
BUFFER_SIZE = 4
OBSERVATION_SIZE = 5
ACTIONS = [0, 1]
ACTION_SIZE = 2
ROUND_TO_DECIMALS = 2

# GAME PARAMETERS
SCREEN_SIZE = WIDTH, HEIGHT = (640, 880)
BACKGROUND = (146, 183, 254)
BIRD_COLOR = (241, 213, 19)
PIPE_COLOR = (44, 176, 26)
FONT = 'dyuthi'


class Actions(enum.Enum):
    Skip = 0
    Fly = 1


class Environment(gym.Env):

    def __init__(self, draw=False, fps=10, debug=False,
                 dist_to_pipe=150, dist_between_pipes=220, obs_this_pipe=False):

        super(Environment, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=OBSERVATION_SIZE * BUFFER_SIZE)
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.pipe_image_up = None
        self.pipe_image_down = None

        if draw:
            pygame.init()
            pygame.display.set_caption('NN FlappyB')

            self.font_game_over = pygame.font.SysFont("ani", 72)
            self.bg = pygame.image.load("flappy_bird_env/assets/bg.png")

            self.pipe_image_up = pygame.image.load(
                "flappy_bird_env/assets/pipe.png")
            self.pipe_image_down = pygame.image.load(
                "flappy_bird_env/assets/pipe_long.png")

        self.dist_between_pipes = dist_between_pipes

        self.fps = fps
        self.debug = debug
        self.draw = draw
        self.dist_to_pipe = dist_to_pipe
        self.obs_this_pipe = obs_this_pipe

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT,
                           PIPE_COLOR, self.dist_between_pipes,
                           self.pipe_image_up, self.pipe_image_down)]

        self.reward = 0
        self.is_done = False
        self.printed_score = False

    def reset(self):
        # resets the whole environment

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT,
                           PIPE_COLOR, self.dist_between_pipes,
                           self.pipe_image_up, self.pipe_image_down)]

        self.reward = 0
        self.is_done = False
        self.printed_score = False

        obs, reward, is_done = self.step(0)

        return obs

    def step(self, action):
        # performs one action onto the environment

        if isinstance(action, np.ndarray):
            if action[0] > action[1]:
                # probability to action
                action = 0
            else:
                action = 1

        obs = []
        rew = 0

        for i in range(BUFFER_SIZE):
            while not self.time_elapsed_since_last_action > self.fps:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1
            o, r, d = self.run_ai_game_step(action)
            rew += r

            for j in range(len(o)):
                obs.append(o[j])

        if rew > 1:
            rew = 1
        elif rew < -1:
            rew = -1
        else:
            rew = 0.1

        obs = np.array(obs)

        return obs, rew, d

    def run_ai_game_step(self, action):

        current_reward = 0.1

        if self.global_time % self.dist_to_pipe == 0:
            self.pipes.append(Pipe(self.screen, WIDTH, HEIGHT,
                                   PIPE_COLOR, self.dist_between_pipes,
                                   self.pipe_image_up, self.pipe_image_down))

        for pipe in self.pipes:
            pipe.update()

            if pipe.off_screen():
                self.pipes.remove(pipe)

            if pipe.hits(self.bird):
                self.game_over()
                current_reward = -1

            if pipe.behind(self.bird):
                self.reward += 1
                current_reward = 1

        self.bird.handle_events_ai(action)
        if self.bird.update():
            self.game_over()
            current_reward = -1

        if self.draw:
            self.screen.blit(self.bg, (0, 0))
            for pipe in self.pipes:
                pipe.draw()
            self.bird.draw(self.reward)
            text = pygame.font.SysFont(FONT, 28).render(
                "SCORE {}".format(self.reward), True, (0, 0, 0))
            self.screen.blit(text, (565 - text.get_width() //
                                    2, 30 - text.get_height() // 2))
            pygame.display.flip()

        obs = self.get_observation_space()

        if self.draw:
            pygame.display.update()

        self.time_elapsed_since_last_action = 0

        return obs, current_reward, self.is_done

    def get_observation_space(self):

        my_pipe = Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR, 220, None, None)
        my_pipe.x = 9999

        if self.obs_this_pipe:
            for pipe in self.pipes:
                if (pipe.x < my_pipe.x) and pipe.x >= (self.bird.x - pipe.width):
                    my_pipe = pipe
        else:
            for pipe in self.pipes:
                if (pipe.x < my_pipe.x) and pipe.x >= (self.bird.x):
                    my_pipe = pipe

        e1 = self.bird.y # position
        e2 = self.bird.vel # bird velocity
        e3 = my_pipe.x - self.bird.x # dist to Pipe
        e4 = my_pipe.top  # top pipe
        e5 = my_pipe.bot  # pipe bot

        if self.draw and self.debug:
            e_d1 = pygame.rect.Rect(self.bird.x, e1, 2, HEIGHT - e1)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d1)

            e_d2 = pygame.rect.Rect(self.bird.x - self.bird.radius,
                                    e2 * 2 + HEIGHT / 2, self.bird.x + self.bird.radius, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d2)

            e_d3 = pygame.rect.Rect(self.bird.x, self.bird.y, e3, 2)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d3)

            e_d4 = pygame.rect.Rect(my_pipe.x - 5, e4, my_pipe.width + 10, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d4)

            e_d5 = pygame.rect.Rect(my_pipe.x - 5, e5, my_pipe.width + 10, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d5)

        e1 = e1 / HEIGHT
        e2 = e2 / self.bird.vel_cap
        e3 = e3 / (WIDTH - 50)
        e4 = e4 / HEIGHT
        e5 = e5 / HEIGHT

        obs = (e1, e2, e3, e4, e5)

        return obs

    def get_action_random(self):
        action = np.random.choice((0, 1), 1, p=(0.45, 0.55))
        return action.item(0)

    def get_observation_size(self):
        return OBSERVATION_SIZE

    def get_observation_size_buffer(self):
        return OBSERVATION_SIZE * BUFFER_SIZE

    def get_actions(self):
        return ACTIONS

    def get_action_size(self):
        return ACTION_SIZE

    def game_over(self):
        if not self.printed_score:
            self.printed_score = True

        if self.draw:
            text = pygame.font.SysFont(FONT, 28).render(
                "Game Over!".format(self.reward), True, (0, 0, 0))
            self.screen.blit(text, (320 - text.get_width() //
                                    2, 240 - text.get_height() // 2))
            pygame.display.flip()
            time.sleep(0.4)
        self.is_done = True