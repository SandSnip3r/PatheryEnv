import gymnasium
import pathery_env
from enum import Enum
import numpy as np
env = gymnasium.make('pathery_env/Pathery-v0', render_mode='ansi')

env.reset()
print(env.render())

# low = np.array([0, 0, 0], dtype=int)
# high = np.array([1, 10, 100], dtype=int)
# space = gymnasium.spaces.Box(low=low, high=high, dtype=int)
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
# testSpace = gymnasium.spaces.MultiDiscrete(np.full((width, height), len(EnumType)))
# # testSpace = gymnasium.spaces.MultiDiscrete(([len(EnumType)]*width)*height)
# print(testSpace)
# print(testSpace.sample())