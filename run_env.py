#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from pathery_env.wrappers.convolution_observation import ConvolutionObservationWrapper
from enum import Enum
import numpy as np

mapString = '27.19.999.Ultra Complex Unlimited...1727020800:,s1.25,r1.,r1.1,r1.14,c1.,z5.,t2.6,f1.,s1.25,r1.,r1.25,f1.,s1.5,z5.5,z5.6,z5.4,r1.1,r1.,r1.10,c5.14,f1.,s1.8,r1.,u4.5,r1.4,c2.4,r1.,r1.4,r1.1,c6.18,f1.,s1.12,r1.6,u1.,u2.4,r1.,r1.6,z5.1,t3.2,t1.4,c3.3,c4.,r1.,t4.,z5.1,f1.,s1.11,c9.13,r1.,r1.1,r1.17,r1.5,f1.,s1.9,r1.4,c7.5,z5.2,c6.1,r1.,r1.25,f1.,s1.3,z5.21,r1.,r1.9,r1.15,f1.,s1.10,c8.,u3.2,r1.10,r1.,r1.9,r1.5,r1.,r1.8,f1.,s1.25,r1.'

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
