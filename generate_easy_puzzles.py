from random import randrange
from block_game import BlockGame
import numpy as np
from PIL import Image

LEFT = 0
RIGHT = 1
CLICK = 2
#os.environ["SDL_VIDEODRIVER"] = "dummy"

def save_image(nparr, file):
  Image.fromarray(nparr).convert("RGB").save(file)


bg=BlockGame(None,False)
for i in range(100):
  bg.reset(None)
  bg.blockWorld.render_cursor=False
  bg.blockWorld.never_render_cursor=True
  steps = randrange(5)
  got_click = False
  for j in range(steps):
    action = randrange(8)
    bg.step(action)
 # if not got_click:
  bg.step(8)
  for k in range(50):
    bg.blockWorld.produce_frame()
  oview = bg.blockWorld.produce_frame()
  np.save("easy_random/" + str(i), oview)
  save_image(oview, "easy_random/"+str(i)+".png")
  