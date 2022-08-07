from block_game import BlockGame
import os
import sys
import pygame
import numpy as np
if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]

cond = np.load(sys.argv[1])
pygame.init()
bg = BlockGame(cond, True)
keymap = {
    pygame.K_LEFT: 0, pygame.K_RIGHT: 1,
    pygame.K_UP: 2, pygame.K_DOWN: 3,
    pygame.K_q: 4, pygame.K_a: 5,
    pygame.K_w: 6, pygame.K_s: 7,
    pygame.K_RETURN: 8
}

while True:
    keys=pygame.key.get_pressed()
    action = None
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN and event.key in keymap:
            action = keymap[event.key]
    if action == None:
        bg.blockWorld.produce_frame()
    else:
        _, score, _, _ = bg.step(action)
        print("Points:" + str(score))