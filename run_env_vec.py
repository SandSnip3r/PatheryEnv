#!/usr/bin/env python

import gymnasium as gym
import pathery_env
from pathery_env.wrappers.action_mask_observation import ActionMaskObservationWrapper
from enum import Enum
import numpy as np

mapString = '6.2.5.Testmap...:,s1.4,f1.,s1.4,f1.'

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
  print(f'Action space: {env.action_space}')

  SEED = 13
  env.action_space.seed(SEED)
  observation, info = env.reset(seed=SEED)

  print('Initial Observations:')
  for ansi in env.call(name="render"):
    print(ansi)
  print('\n')

  for i in range(2):
    actionSample = env.action_space.sample()
    print(f'Going to take actions:\n{actionSample}')

    nextObservation, reward, terminated, truncated, info = env.step(actionSample)
    # print(f'nextObservation: {nextObservation}')
    # print(f'reward: {reward}')
    # print(f'terminated: {terminated}')
    # print(f'truncated: {truncated}')
    # print(f'info: {info}')

    finalBoards = nextObservation['board']
    
    finalObservationMaskKey = '_final_observation'
    finalObservationKey = 'final_observation'
    if finalObservationMaskKey in info:
      print(f'Saw a final observation')
      finalObsMask = info[finalObservationMaskKey]
      print(f'Mask: {type(finalObsMask)},{finalObsMask.shape}: {finalObsMask}')
      # print(f'Final obs: {info[finalObservationKey]}')
      for i in range(len(finalObsMask)):
        if finalObsMask[i]:
          finalBoards[i] = info[finalObservationKey][i]['board']
      print(f'Final boards {type(finalBoards)}: {finalBoards}')
      # nextBoards = np.where(finalObsMask[:, np.newaxis, np.newaxis, np.newaxis], finalBoards, nextObservation['board'])

    # print(f'{observation["board"]}\n->\n{nextObservation["board"]}')


    # print(f'tmp: {tmp}')


    # print(f'Obs: {type(observation)}')
    # print(f'Obs: {type(observation.items())}')
    # for i in observation.items():
    #   print(i)
    # print(f'Obs: {observation["board"]}')

    print(f'Got rewards:\n{reward}')
    print('New observations:')
    for ansi in env.call(name="render"):
      print(ansi)
    print(f'\n{"="*50}\n')
