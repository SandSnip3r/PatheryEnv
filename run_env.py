#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from pathery_env.wrappers.convolution_observation import ConvolutionObservationWrapper
from enum import Enum
import numpy as np

mapString = '7.6.8.Simple...:,s1.,u1.4,t1.7,u1.20,t1.5,f1.'

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
  env = ConvolutionObservationWrapper(env)

  while True:
    obs, info = env.reset()
    done = False

    if isWrappedBy(env, ActionMaskObservationWrapper):
      print(f'Mask; {obs["action_mask"]}')
    print(f'Start; {info}')

    def readPair():
      user_input = input("Enter two integers separated by space: ")
      num1, num2 = user_input.split()
      return (int(num1), int(num2))

    while not done:
      print(env.render())
      pairInput = readPair()
      while pairInput not in env.action_space:
        print('invalid action')
        pairInput = readPair()
      observation, reward, done, _, info = env.step(pairInput)
      print(f'Reward: {reward}, info: "{info}"')
      if done:
        print(env.render())
        print(f'\n{"~"*20}End of episode{"~"*20}\n')
