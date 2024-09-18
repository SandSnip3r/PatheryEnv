#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from pathery_env.wrappers.flatten_action import FlattenActionWrapper
from enum import Enum
import numpy as np

mapString = '4.3.3.Simple...1725768000:,s1.7,f1.'

def isWrappedBy(env, wrapper_type):
  """Recursively unwrap env to check if any wrapper is of type wrapper_type."""
  current_env = env
  while isinstance(current_env, gym.Wrapper):
    if isinstance(current_env, wrapper_type):
      return True
    current_env = current_env.env  # Unwrap to the next level
  return False

if __name__ == "__main__":
  # env = gym.make('pathery_env/Pathery-RandomNormal', render_mode='ansi')
  env = gym.make('pathery_env/Pathery-FromMapString', render_mode='ansi', map_string=mapString)

  while True:
    obs, info = env.reset()
    done = False

    if isWrappedBy(env, ActionMaskObservationWrapper):
      print(f'Mask; {obs["action_mask"]}')
    print(f'Start; {info}')

    def readAction():
      def pairToAction(row, col):
        height, width = env.unwrapped.gridSize
        return col + row * width

      user_input = input("Enter two integers separated by space: ")
      num1, num2 = user_input.split()
      x = np.asarray([pairToAction(int(num1), int(num2))], dtype=np.int32)
      return x

    while not done:
      print(env.render())
      userAction = readAction()
      while userAction not in env.action_space:
        print(f'Action {userAction} is invalid. It must fit in {env.action_space}')
        userAction = readAction()
      observation, reward, terminated, truncated, info = env.step(userAction)
      print(f'Reward: {reward}, info: "{info}"')
      done = terminated or truncated
      if done:
        print(env.render())
        print(f'\n{"~"*20}End of episode{"~"*20}\n')
