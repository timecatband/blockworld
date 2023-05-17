import Box2D
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import draw_world
# Define constants
NUM_BLOCKS = 5
MAX_TRIES = int(sys.argv[2])
WIDTH = 8
HEIGHT = 8

# Create Box2D world
world = Box2D.b2World()
ground_shape = Box2D.b2PolygonShape(box=(WIDTH*2, 1))  # 1 unit high, and as wide as the width of your world
ground_body = world.CreateStaticBody(position=(WIDTH, 0), shapes=ground_shape)  # position the ground at the bottom of your world


# Initialize dataset
data = []


def generate_random_block(width, height, stack):
    # Generate random size within given limits
    size = np.random.uniform(1.0, 3.0, 2)  # random size between 0.5 and 1.5

    if stack:  # if there is at least one block in the stack
        # Position the block on top of the topmost block in the stack
        top_block, last_pos, last_size = stack[-1]
        pos_x, pos_y = last_pos
        pos_y += last_size[1]+size[1]/2  # add half the height of the block to the y position
        pos_x += np.random.uniform(-last_size[0], last_size[0])*1.4  # add a random offset to the x position
    else:
        # Position the block at a random x position and just above the ground
        pos_x = np.random.uniform(0, width)  # random x position within width
        pos_y = 1 + size[1] / 2  # just above the ground
    pos = [pos_x, pos_y]
    
    # Create the block
    block_shape = Box2D.b2PolygonShape(box=(size[0]/2, size[1]/2))  # half-width, half-height for box
    
    # Add the block to the world
    block_body = world.CreateDynamicBody(position=Box2D.b2Vec2(*pos), shapes=block_shape)
    box = block_body.CreatePolygonFixture(box=(size[0]/2, size[1]/2), density=1, friction=0.3)
    
    return block_body, pos, size/2


def check_stability(stack):
    # Check if any block is moving
    for block,_,_ in stack:
        if block.linearVelocity.length > 0.01:  # if the block is moving
            return False  # the stack is not stable
    
    return True  # the stack is stable

def average_velocity(stack):
    avg = Box2D.b2Vec2(0,0)
    for block,_,_ in stack:
        avg += block.linearVelocity
    return avg

def encode_stack(stack):
    # Convert the stack to a suitable representation for machine learning
    # This could be a simple list of positions and sizes, or something more complex
    return [(pos[0], pos[1], size[0], size[1]) for block,pos,size in stack]


output_dir = sys.argv[1]

for i in range(MAX_TRIES):
    # Clear the world
    for body in world.bodies:
        if body == ground_body:
            continue
        world.DestroyBody(body)
    
    # Initialize a new stack
    stack = []
    
    for _ in range(NUM_BLOCKS):
        # Generate a new block with random size and position
        block, initial_size, initial_pos = generate_random_block(WIDTH, HEIGHT, stack)
        
        # Add the block to the stack
        stack.append((block, initial_size, initial_pos))
    
    if i < 50:
        draw_world(world, f"{output_dir}/begin_{i}.png")  # draw initial state
    
    # Simulate the physics
    integrated_velocity = Box2D.b2Vec2(0,0)
    for _ in range(100):  # simulate for some time
        world.Step(1.0/30.0, 10, 10)
        integrated_velocity += average_velocity(stack)
    
    # Check if the stack is stable
    is_stable = check_stability(stack)
#    print("Average velocity:", integrated_velocity/1000)
 #   print("Stable:", is_stable)

    if i < 50:
        draw_world(world, f"{output_dir}/end_{i}.png")  # draw final state
    if i % 1000 == 0:
        print("Iteration:", i)
    
    # Add the result to the dataset
    data.append((encode_stack(stack), (integrated_velocity[0], integrated_velocity[1])))

# Save the data as an npy file
np.save(f'{output_dir}/data.npy', data)
