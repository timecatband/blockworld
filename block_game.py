import time 
import os, sys

import pygame
import torch
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_RETURN)

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

import numpy as np
import torch.nn

from PIL import Image
import IPython.display

SCREEN_WIDTH=128
SCREEN_HEIGHT=128
NUM_ACTIONS=9
PPM = SCREEN_WIDTH/10.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 20.0 / TARGET_FPS
BOX_FRICTION=0.8
SIMULATE_STEPS=8
MOVEMENT_MULTIPLIER=0.5

#os.environ["SDL_VIDEODRIVER"] = "dummy"

def save_tensor_image(t, file):
  t = t.cpu().permute(1,2,0)*255
  t = t.byte()
  Image.fromarray(np.array(t)).convert("RGB").save(file)


def display_image(nparr, file):
  Image.fromarray(nparr).convert("RGB").save(file)

class BlockWorld:
  colors = {
    staticBody: (255, 0, 0, 255),
    dynamicBody: (0, 255, 0, 255),
  }
  
  def __init__(self):
    self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Blockworld')
    self.clock = pygame.time.Clock()
    self.world = world(gravity=(0, -1), doSleep=True)
    ground_body = self.world.CreateStaticBody(
      position=(0, -5),
      shapes=polygonShape(box=(10, 5)),
    )
    self.cursor_pos=[5,5,1,1]
    self.draw_cursor = True
    self.videoOut = None
    self.never_render_cursor=False
  def add_block(self, position, size):
    position = (position[0],SCREEN_HEIGHT/PPM-position[1])
    dynamic_body = self.world.CreateDynamicBody(position=position)
    box = dynamic_body.CreatePolygonFixture(box=size, density=0.1, friction=BOX_FRICTION)
  def record(self, videoOut):
    self.videoOut = videoOut

  def cursor_left(self):
    self.cursor_pos[0] = max(self.cursor_pos[0]-MOVEMENT_MULTIPLIER, 0)
  def cursor_right(self):
    self.cursor_pos[0] = min(self.cursor_pos[0]+MOVEMENT_MULTIPLIER, SCREEN_WIDTH)
  def cursor_up(self):
    self.cursor_pos[1] = max(self.cursor_pos[1]-MOVEMENT_MULTIPLIER, 0)
  def cursor_down(self):
    self.cursor_pos[1] = min(self.cursor_pos[1]+MOVEMENT_MULTIPLIER, SCREEN_HEIGHT-30)
  def cursor_w_up(self):
    self.cursor_pos[2]=min(self.cursor_pos[2]+MOVEMENT_MULTIPLIER,5)
  def cursor_w_down(self):
    self.cursor_pos[2] = max(self.cursor_pos[2]-MOVEMENT_MULTIPLIER, 2)
  def cursor_h_up(self):
    self.cursor_pos[3]=min(self.cursor_pos[3]+MOVEMENT_MULTIPLIER,5)
  def cursor_h_down(self):
    self.cursor_pos[3] = max(self.cursor_pos[3]-MOVEMENT_MULTIPLIER, 2)
  def click_cursor(self):
    self.add_block((self.cursor_pos[0]+self.cursor_pos[2]/2, self.cursor_pos[1]),
                    (self.cursor_pos[2]/2, self.cursor_pos[3]/2))
  def act(self, action):
        actions = [self.cursor_left, self.cursor_right, self.cursor_up, self.cursor_down,
            self.cursor_w_up, self.cursor_w_down, self.cursor_h_up, self.cursor_h_down, self.click_cursor]
        actions[action]()
  def produce_frame(self):
    self.screen.fill((0, 0, 0, 0))
    # Draw the world

    for body in self.world.bodies:
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) * PPM for v in shape.vertices]
            #pygame is upside-down from box2d
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, self.colors[body.type], vertices)
    if self.draw_cursor == True and not self.never_render_cursor:
      pygame.draw.rect(self.screen, (0,0,255,255), [i*PPM for i in self.cursor_pos])


    self.world.Step(TIME_STEP, 10, 10)

    pygame.display.flip()
    view = pygame.surfarray.array3d(self.screen)
    if (self.videoOut != None):
      self.videoOut.write(view.transpose([1,0,2]))
    return view
    #return view.transpose([1,0,2])


class BlockGame():
  def __init__(self, target=None, use_target = False, device="cpu"):
    self.blockWorld = BlockWorld()
    self.target = target
    self.device = device
    self.use_target = use_target
    if (use_target == True):
      self.tensor_target = torch.tensor(self.target).permute(2,0,1).float().to(device)/255
      self.max_diff = (self.tensor_target-self.get_screen()).sum()

    self.blockWorld.draw_cursor = False
    # Check in to this if changing base world
    self.blockWorld.draw_cursor = True
    self.blocksAdded = 0
  def reset(self, target):
    self.blockWorld = BlockWorld()
    self.blocksAdded = 0
    self.target = target
    if (self.use_target):
      self.tensor_target = torch.tensor(self.target).permute(2,0,1).float().to(self.device)/255
  def score(self, target, frame):
   # save_tensor_image(target, "target.png")
   # save_tensor_image(frame, "frame.png")
    current_diff = abs(float((target-frame).sum()))
    return current_diff/self.max_diff
  def get_target(self):   
    return self.tensor_target
  def step(self, action):
    x = action
    if (self.use_target == True):
     self.blockWorld.draw_cursor = False
     startScore = self.score(self.tensor_target, self.get_screen())
    self.blockWorld.act(action)
    lastView = None
    steps = 1
    if x == 2:
      steps = SIMULATE_STEPS
    
    for i in range(steps):
      lastView = self.blockWorld.produce_frame()
    if self.use_target != True:
      return None
    endScore = self.score(self.tensor_target,self.get_screen())
    finalScore = startScore-endScore
    if (endScore <= 1.0/255):
      if x == 2:
        finalScore *= 2
    self.blockWorld.draw_cursor = True
    return self.get_screen(), finalScore, False if endScore != 0 else True, {}
    #return self.get_screen(), finalScore, False if self.blocksAdded < 4 else True, {}
  def record(self, videoOut):
    self.blockWorld.record(videoOut)
  def get_screen(self):
    return torch.tensor(self.blockWorld.produce_frame().astype("float32")).permute(2,0,1)/255



