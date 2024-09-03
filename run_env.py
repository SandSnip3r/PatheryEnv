#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from enum import Enum
import numpy as np

mapString = '24.16.10.EasyAndHard...:33,s1.14,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.278,r1.10,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.,r1.1,f1.,c1.'

if __name__ == "__main__":
  # env = gym.make('pathery_env/Pathery-RandomNormal', render_mode='ansi')
  env = gym.make('pathery_env/Pathery-FromMapString', render_mode='ansi', map_string=mapString)

  obs, info = env.reset()
  done = False
  print(f'Start; {info}')
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
