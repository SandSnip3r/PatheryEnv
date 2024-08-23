import gymnasium as gym
import pathery_env
from enum import Enum
import numpy as np
env = gym.make('pathery_env/Pathery-v0', render_mode='ansi')

env.reset()
done = False

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
