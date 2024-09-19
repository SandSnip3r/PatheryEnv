#!/usr/bin/env python

import gymnasium as gym
import numpy as np

if __name__ == "__main__":
  observations = np.random.randint(0,100,(2,5))
  print(f'observations:\n{observations}\n')
  mask = np.random.randint(0,2,(2,),dtype=np.bool)
  print(f'mask:\n{mask}\n')
  final_observations = np.random.randint(0,100,(2,5))
  print(f'final_observations:\n{final_observations}\n')

  new = np.where(mask[:, np.newaxis], observations, final_observations)
  print(f'new:\n{new}\n')

