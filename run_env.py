import gymnasium as gym
import pathery_env
from enum import Enum
import numpy as np
env = gym.make('pathery_env/Pathery-v0', render_mode='ansi')

env.reset()
print(env.render())

terminated = False
while not terminated:
  reward = -1
  while reward < 0:
    action = env.action_space.sample()
    _, reward, terminated, _, _ = env.step(action)
  print(env.render())
  # return observation, reward, terminated, False, info

# low = np.array([0, 0, 0], dtype=int)
# high = np.array([1, 10, 100], dtype=int)
# space = gym.spaces.Box(low=low, high=high, dtype=int)
# for _ in range(100):
#   print(space.sample())


# class EnumType(Enum):
#   OPEN = 0
#   BLOCKED_PRE_EXISTING = 1
#   BLOCKED_PLAYER_PLACED = 2
#   START = 3
#   GOAL = 4

# width = 4
# height = 5
# testSpace = gym.spaces.MultiDiscrete(np.full((width, height), len(EnumType)))
# # testSpace = gym.spaces.MultiDiscrete(([len(EnumType)]*width)*height)
# print(testSpace)
# print(testSpace.sample())

# x = gym.spaces.MultiDiscrete((10,2))
# print(x.sample())
# print(x.sample())