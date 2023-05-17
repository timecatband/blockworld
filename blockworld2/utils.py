import matplotlib.pyplot as plt
import Box2D
import numpy as np

WIDTH = 8
HEIGHT = 8

def draw_world(world, filename):
    plt.figure(figsize=(10,10))
    for body in world.bodies:
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) for v in shape.vertices]
            vertices = [(v[0], v[1]) for v in vertices]
            plt.gca().add_patch(plt.Polygon(vertices, edgecolor='black', fill=None))

    plt.xlim(-WIDTH, 2 * WIDTH)
    plt.ylim(-HEIGHT, 2 * HEIGHT)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filename)
    plt.close()

def draw_world_torch(world, filename):
    print("World shape:" + str(world.shape))
    plt.figure(figsize=(10,10))
    for i in range(world.shape[0]):
        v = world[i].tolist()
        x,y,w,h = v[0], v[1], v[2], v[3]
        w *= 2 # lol
        h *= 2
        print("x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
        vertices = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2), (x-w/2, y+h/2)]
        plt.gca().add_patch(plt.Polygon(vertices, edgecolor='black', fill=None))

    plt.xlim(-WIDTH, 2 * WIDTH)
    plt.ylim(-HEIGHT, 2 * HEIGHT)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filename)
    plt.close()