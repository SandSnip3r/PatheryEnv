import gymnasium
import pathery_env
import numpy as np
env = gymnasium.make('pathery_env/Pathery-v0', render_mode='ansi')

env.reset()
print(env.render())

# low = np.array([0, 0, 0], dtype=int)
# high = np.array([1, 10, 100], dtype=int)
# space = gymnasium.spaces.Box(low=low, high=high, dtype=int)
# for _ in range(100):
#   print(space.sample())