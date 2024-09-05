#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from pathery_env.wrappers.convolution_observation import ConvolutionObservationWrapper
from enum import Enum
import numpy as np

# mapString = '24.16.10.EasyAndHard...:33,s1.14,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.278,r1.10,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.1,f1.,c1.'
mapString = '17.9.14.Normal...1725422400:,r3.8,r1.6,f1.,r3.15,f1.,s1.3,r1.5,r1.5,f1.,r3.15,f1.,r3.8,c2.6,f1.,r3.3,r1.4,r1.6,f1.,r3.,r1.8,c1.5,f1.,r3.2,r1.3,r1.,r1.7,f1.,r3.1,r1.13,f1.'

if __name__ == "__main__":
  # env = gym.make('pathery_env/Pathery-RandomNormal', render_mode='ansi')
  env = gym.make('pathery_env/Pathery-FromMapString', render_mode='ansi', map_string=mapString)
  env = ActionMaskObservationWrapper(env)
  env = ConvolutionObservationWrapper(env)

  while True:
    obs, info = env.reset()
    done = False

    print(f'Start; {info}')
    if isinstance(env, ActionMaskObservationWrapper):
      # TODO: Does not work when Conv is applied after ActionMask wrapper (but the action mask is in the observation)
      print(f'Mask; {obs["action_mask"]}')

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
