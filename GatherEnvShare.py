
import random

import os, sys
with open(os.devnull, 'w') as f:
    # disable stdout
    oldstdout = sys.stdout
    sys.stdout = f

    import pygame
    from pygame import *

    # enable stdout
    sys.stdout = oldstdout


import math # needs to be imported after pygame
import time

import gym
from gym import spaces, logger
import numpy as np


class Simulation(gym.Env):

    # stuff for pygame or something
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    # should work hopefully on random generated rounds
    def seed(self, seed=-1, everygame_same=False):
        if(everygame_same):
            self.seeding = seed
        else:
            random.seed(seed)

    # inits
    def __init__(self):

        # setup for the gameplay
        self.seeding = -1
        self.initialGameTime = 4
        self.gameTime        = 0
        self.highscore       = 0
        self.WHITE = (255, 255, 255)
        self.ORANGE = (255,140,0)
        self.GREEN = (100, 255, 100)
        self.BLACK = (0, 0, 0)

        self.WIDTH = 64
        self.HEIGHT = self.WIDTH
        self.player = [0, 0]
        self.player_vel = 1.6
        self.score = 0
        self.enemies = []
        self.initial_enemies = 300
        self.area_in_each_direction = 400
        self.desired_pos = [self.WIDTH/2, self.HEIGHT/2]
        self.tickspeed = 60

        # pygame setup
        pygame.init()
        pygame.display.set_caption('Game')
        self.fps = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), 0, 32)

        # variables used by tensorflow, when accessing gym
        self.observation_space = spaces.Box(low=0, high=1, shape=np.array(
            (self.WIDTH, self.HEIGHT)), dtype=np.uint8)
        self.action_space = spaces.Discrete(10)


    ## resets the game state and returns
    def reset(self):
        if(self.seeding!=-1):
            random.seed(self.seeding)
        self.gameTime=self.initialGameTime

        self.enemies.clear()
        self.score = 0
        self.player = [self.WIDTH // 2, self.HEIGHT // 2]
        self.desired_pos = self.player

        for i in range(self.initial_enemies):
            self.enemies.append([
                random.randint(
                    self.WIDTH//2-self.area_in_each_direction,
                    self.WIDTH//2+self.area_in_each_direction),
                random.randint(
                    self.HEIGHT//2-self.area_in_each_direction,
                    self.HEIGHT//2+self.area_in_each_direction), 20])
        self.render()
        return self.get_screenshot()


    # action gets passed (either number 0 - 9) or one hot encoded version
    # 0 does nothing, 9 makes agent stand ground, 1-8 moves directions
    def _handle_action(self, action):

        if isinstance(action, np.ndarray):
            if(action.shape is not ()):
                action = np.argmax(action, axis=0)

        mx = self.WIDTH//2
        my = self.HEIGHT//2

        range = 20

        if(action==9): # stand ground
            self.desired_pos = self.player
        if(action==0): # go to last desired direction
            return
        if(action==1): # right
            self.request_newpos([mx+range+range/2,my])
        if(action==2): # left
            self.request_newpos([mx-range-range/2,my])
        if(action==3): # down
            self.request_newpos([mx,my+range+range/2])
        if(action==4): # up
            self.request_newpos([mx,my-range-range/2])
        if(action==5): # down right
            self.request_newpos([mx+range,my+range])
        if(action==6): # up right
            self.request_newpos([mx+range,my-range])
        if(action==7): # down left
            self.request_newpos([mx-range,my+range])
        if(action==8): # up left
            self.request_newpos([mx-range,my-range])


    # makes the game update a step, according to action
    # takes input from handle_events() or the ML algo
    def step(self, action=0):
        # print (action)
        # time.sleep(0.01) # 100 fps, uncomment to have it at "normal" speed

        done = False
        if(self.gameTime<=0):
            done = True
            return np.array(self.get_screenshot()), self.score, done,{}

        self._handle_action(action)
        self._handle_events() # handles mouseinput, overwriting requested action

        if(self.score>self.highscore):
            self.highscore = self.score

        self.gameTime -=0.015


        dist = math.hypot(self.desired_pos[0] - self.player[0],
                          self.desired_pos[1] - self.player[1])

        if(dist<2):
            self.desired_pos = self.player
        else:
            angle = math.atan2(self.player[1]-self.desired_pos[1],
                               self.player[0]-self.desired_pos[0])
            self.player[0]+= -self.player_vel * float(math.cos(angle))
            self.player[1]+= -self.player_vel * float(math.sin(angle))

        self._handle_enemies(action)

        # self.score-=0.1

        self.render()
        scorebuffer = self.score
        self.score = 0
        return np.array(self.get_screenshot()), scorebuffer, done,{}

    def _handle_enemies(self, action):
        for enemy in self.enemies:

            enemy_dist = math.hypot(enemy[0] - self.player[0],
                                    enemy[1] - self.player[1])

            #if(dist<2): # life should only be reduced when stand ground, but thats even harder
            # if(enemy_dist<15):
            if(enemy_dist<5):
                self.enemies.remove(enemy)
                self.score+=.25 # hit enemy
                if(enemy[2]<=0):
                    self.score+=2. # kill enemy
                    #print(get_enemy_positions())
                else:
                    self.enemies.append([enemy[0],enemy[1],enemy[2]-1])
            # else:
            #     if(action==0 or action==9):
            #         self.score-=0.5


    # draws the individual things on the screen
    # (player, enemies, move indicator, points)
    def _draw(self):

        # if the env is closed forceably
        # self.screen.fill(self.BLACK) will trigger an error,
        # "pygame.error: display Surface quit" will be triggered,
        # but that should not be a problem for ML learning/testing
        self.screen.fill(self.BLACK)

        player = self.player
        gameTime = self.gameTime
        desired_pos = self.desired_pos
        score = self.score
        highscore = self.highscore

        def draw_player():
            # pygame.draw.circle(self.screen, self.ORANGE, [self.WIDTH // 2, self.HEIGHT // 2], 15, 0)
            pygame.draw.rect(self.screen, self.ORANGE, (self.WIDTH // 2, self.HEIGHT // 2, 4, 4))

        def draw_pointer():
            # pygame.draw.circle(self.screen, self.WHITE, [int(desired_pos[0] - player[0] + self.WIDTH // 2), int(desired_pos[1] - player[1] + self.HEIGHT // 2)], 9, 0)
            pygame.draw.rect(self.screen, self.WHITE, (int(desired_pos[0] - player[0] + self.WIDTH // 2), int(desired_pos[1] - player[1] + self.HEIGHT // 2), 4, 4))

        def draw_enemies():
            for enemy in self.enemies:
                relative = self._absolute_to_relative(enemy)
                # pygame.draw.circle(self.screen, self.GREEN, relative, 10, 0)
                pygame.draw.rect(self.screen, self.GREEN,( relative[0],relative[1], 4, 4))

                # # draw healthbars
                # pygame.draw.line(self.screen, self.GREEN,
                #                  [relative[0]-11, relative[1]-15],
                #                  [relative[0]-11 +enemy[2], relative[1]-15],
                #                  4)

        def draw_gameinfo():
            font = pygame.font.SysFont("Arial", 30)

            label1 = font.render("Time " + str(int(gameTime)), 1, (255, 255, 0))
            self.screen.blit(label1, (50, 15))

            label2 = font.render("Score " + str(score), 1, (255, 255, 0))
            self.screen.blit(label2, (440, 15))

            label2 = font.render("Highscore " + str(highscore), 1, (255, 255, 0))
            self.screen.blit(label2, (440, 45))

        #draw_gameinfo()
        draw_player()
        # draw_pointer()
        draw_enemies()


    # if you want to know (unused helper method)
    def _absolute_to_relative(self,enemy):
        absolute = [int(enemy[0]-self.player[0])+ self.WIDTH//2, int(enemy[1]-self.player[1])+ self.HEIGHT//2]
        return absolute

    # get all enemy positions [[x,y],[x,y]...] format
    # if you want to know them
    def get_enemy_positions(self):
        lst = []
        for enemy in self.enemies:
            lst.append(self._absolute_to_relative(enemy))
        return lst


    #requesting a new position to walk to [x,y]
    def request_newpos(self, pos):

        self.desired_pos = [pos[0]  + self.player[0] - (self.WIDTH//2 ) ,
                            pos[1]  + self.player[1] - (self.HEIGHT//2) ]



    # takes input from human and give it to handle_events()
    def keyup(self, event):
        if event.key == K_UP:
            self.tickspeed +=30
        elif event.key == K_DOWN:
            self.tickspeed -=30
            if(self.tickspeed<=1):
                self.tickspeed=5


    # handles human input, useless (and unused) for ML argos
    def _handle_events(self):
        for event in pygame.event.get():

            if event.type == KEYUP:
                self.keyup(event)

            if(event.type == pygame.MOUSEBUTTONDOWN):

                pos = pygame.mouse.get_pos()
                self.request_newpos(pos)

            elif event.type == QUIT:
                sys.exit() # quits program if you press x


    # no idea what is supposed to be returned,
    # but almost no algorithm uses the returnvalue anyway,
    # renders the stuff onto the pygame.Surface
    def render(self, mode='human', close=None):

        self._draw()
        pygame.display.update()
        self.fps.tick()
        return self.get_screenshot()


    # make screenshot and return it as a numpy array
    def get_screenshot(self):
        image = pygame.Surface((self.WIDTH, self.HEIGHT))  # Create image surface
        image.blit(self.screen, (0, 0), ((0,0), (self.WIDTH, self.HEIGHT)))  # Blit portion of the display to the image

        # imgdata = pygame.surfarray.array2d(image)
        imgstring = pygame.image.tostring(image, "RGBA",False)
        # pygame.image.save(image, "screenshot.jpg")  # Save the image to the disk
        # nparr = np.fromstring(imgstring,np.uint8) # raised a warning
        nparr = np.frombuffer(imgstring,np.uint8)
        dimnparr = nparr.reshape((self.HEIGHT,self.WIDTH,4)).astype("uint8")

        gray = np.dot(dimnparr[...,:3], [0.299, 0.587, 0.114])

        normalized = np.dot(gray, 1/255)


        # from PIL import Image
        # ymage = Image.fromarray(normalized, "RGB")
        # ymage.show()
        # time.sleep(1)


        # IF YOU CHANGE THE RETURN, ALSO CHANGE THE OBSERVATION SPACE
        return normalized

    def close(self):
        pass

    # this method can be used to play as a human, if True
    # if False the game plays itself only using random actions
    # a ML algo could be inserted here, but for OO reasons in an extra file
    # if not human, random actions
    def start(self, human=True):

        counter = 0

        self.seed(1)
        env = self
        observation = env.reset()
        frame = 0

        done = False
        while not done:
            frame+=1
            # env.render()
            # print(observation)
            if(not human):
                action = env.action_space.sample()
                # action = int(input()) # playing with numeric inputs
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step()
                # print(reward)
            print(reward)

            time.sleep(0.02) # 50 fps, uncomment to have it at "normal" speed

            if done:
                # print(env.score)
                env.reset()
                done = False
                counter+=1
                print(counter) # game counter
                # env.close() # plays only 1 round and closes if uncommented

# gym.register(id="gatherer-v0",
#              max_episode_steps=500,
#              entry_point="mystuff_q.environments.GatherEnvironment:Simulation"
#              )

# print("imported simulation")

if __name__ == "__main__":
    Simulation().start(False) # uncomment to play the game as human, by starting this file

