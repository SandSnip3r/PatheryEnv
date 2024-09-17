#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from pathery_env.wrappers.convolution_observation import ConvolutionObservationWrapper
from enum import Enum
import numpy as np

mapString = '3.3.3.Simple...1725768000:,s1.7,f1.'

def isWrappedBy(env, wrapper_type):
  """Recursively unwrap env to check if any wrapper is of type wrapper_type."""
  current_env = env
  while isinstance(current_env, gym.Wrapper):
    if isinstance(current_env, wrapper_type):
      return True
    current_env = current_env.env  # Unwrap to the next level
  return False

if __name__ == "__main__":
  # env = gym.make_vec('pathery_env/Pathery-RandomNormal', num_envs=2, vectorization_mode="sync", render_mode='ansi')
  env = gym.make_vec('pathery_env/Pathery-FromMapString', num_envs=2, vectorization_mode="sync", render_mode='ansi', map_string=mapString)
  # env = ConvolutionObservationWrapper(env)

  SEED = 12
  env.action_space.seed(SEED)
  observation, info = env.reset(seed=SEED)

  print('Initial Observations:')
  for ansi in env.call(name="render"):
    print(ansi)
  print('\n')

  for i in range(2):
    actionSample = env.action_space.sample()
    print(f'Going to take actions:\n{actionSample}')

    observation, reward, terminated, truncated, info = env.step(actionSample)
    # print(f'observation: {observation}')
    # print(f'reward: {reward}')
    # print(f'terminated: {terminated}')
    # print(f'truncated: {truncated}')
    # print(f'info: {info}')

    print(f'Got rewards:\n{reward}')
    print('New observations:')
    for ansi in env.call(name="render"):
      print(ansi)
    print(f'\n{"="*50}\n')
