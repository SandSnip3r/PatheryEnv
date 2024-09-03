from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque

class InternalCellType(Enum):
  OPEN = 0
  ROCK = 1
  WALL = 2
  START = 3
  GOAL = 4

class CellType(Enum):
  OPEN = 0
  BLOCKED = 1
  START = 2
  GOAL = 3

def createRandomNormal(render_mode):
  return PatheryEnv.randomNormal(render_mode)

def fromMapString(render_mode, map_string):
  return PatheryEnv.fromMapString(render_mode, map_string)

class PatheryEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 4}

  @classmethod
  def randomNormal(cls, render_mode):
    return cls(render_mode=render_mode)

  @classmethod
  def fromMapString(cls, render_mode, map_string):
    return cls(render_mode=render_mode, map_string=map_string)

  def __init__(self, render_mode, map_string=None):
    self.random_map = (map_string == None)

    self.startPositions = []
    self.goalPositions = []
    self.rocks = []
    self.checkpoints = []

    if map_string is not None:
      self._initializeFromMapString(map_string)
    else:
      # Size and wall count are hard coded for random maps
      self.gridSize = (9, 17)
      self.wallsToPlace = 14
      self.maxCheckpointCount = 2

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.Dict({
      'board': spaces.MultiDiscrete(np.full((self.gridSize[0], self.gridSize[1]), len(CellType) + self.maxCheckpointCount)),
      'action_mask': spaces.Box(low=0, high=1, shape=(self.gridSize[0], self.gridSize[1]), dtype=np.int8)
    })

    # Possible actions are which 2d position to place a wall in
    self.action_space = spaces.MultiDiscrete((self.gridSize[0], self.gridSize[1]))

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Reset data
    self._resetGrid()
    self.rewardSoFar = 0

    # Set the number of walls that the user can place
    self.remainingWalls = self.wallsToPlace

    if self.random_map:
      # Reset data
      self.startPositions = []
      self.goalPositions = []
      self.rocks = []
      self.checkpoints = []

      # Choose a random start along the left edge
      randomStartPos = (self.np_random.integers(low=0, high=self.gridSize[0], dtype=np.int32), 0)
      self.startPositions.append(randomStartPos)

      # All other cells on the left edge must be a rock
      for row in range(self.gridSize[0]):
        if row != randomStartPos[0]:
          self.rocks.append((row, 0))

      # For normal puzzles, every cell on the right edge is a goal
      for row in range(self.gridSize[0]):
        self.goalPositions.append((row, self.gridSize[1]-1))

      # Pick checkpoints
      self._generateRandomCheckpoints(checkpointCount=self.maxCheckpointCount)

    # Place the start(s)
    for startPos in self.startPositions:
      self.grid[startPos[0]][startPos[1]] = InternalCellType.START.value

    # Place the goal(s)
    for goalPos in self.goalPositions:
      self.grid[goalPos[0]][goalPos[1]] = InternalCellType.GOAL.value

    # Place checkpoints
    for row, col, cellValue in self.checkpoints:
      self.grid[row][col] = cellValue

    # Place rocks
    for rockPos in self.rocks:
      self.grid[rockPos[0]][rockPos[1]] = InternalCellType.ROCK.value

    # Finally, random rock placement must be done after everything else has been placed so that we can check that no rock blocks any path
    if self.random_map:
      # Pick rocks
      self._generateRandomRocks(rocksToPlace=14)

    # Keep track of path length
    self.lastPathLength = self._calculateShortestPath()

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def step(self, action):
    if self.grid[action[0]][action[1]] == InternalCellType.OPEN.value:
      self.grid[action[0]][action[1]] = InternalCellType.WALL.value
      self.remainingWalls -= 1
    else:
      # Invalid position; reward is -1, episode terminates
      raise ValueError(f'Invalid action {action}')

    pathLength = self._calculateShortestPath()

    if pathLength == 0:
      # Blocks path; reward is -1 for entire episode, episode terminates
      self.lastPathLength = 0
      return self._get_obs(), -self.rewardSoFar-1, True, False, self._get_info()

    terminated = self.remainingWalls == 0
    reward = pathLength - self.lastPathLength
    self.rewardSoFar += reward
    if reward < 0:
      raise ValueError(f'Reward is negative: {reward}')
    self.lastPathLength = pathLength

    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "ansi":
      return self._render_ansi()

  def close(self):
    pass

  # =========================================================================================
  # ================================ Private functions below ================================
  # =========================================================================================

  def _linearTo2d(self, pos):
    return pos//self.gridSize[1], pos%self.gridSize[1]

  def _initializeFromMapString(self, map_string):
    self.maxCheckpointCount = 0
    # Map string format
    # <width;int>.<height;int>.<num walls;int>.<name;string>...<unk>:([<number of open cells>],<cell type>.)*
    # Cell types:
    #   r1: Map-pre-placed rock
    #   r2: Player-placed wall
    #   r3: Map boundary rock
    #   s1: Start (1st path)
    #   s2: Start (2nd path)
    #   f1: Finish/goal
    #   c[0-9]+: Checkpoint
    metadata, map = map_string.split(':')
    width, height, numWalls, *_ = metadata.split('.')
    # Get size and wall count from map string
    self.gridSize = (int(height), int(width))
    self.wallsToPlace = int(numWalls)
    # Save rocks, start(s), goal(s), and checkpoint(s) from map string
    mapCells = map.split('.')
    currentIndex = -1
    for cell in mapCells:
      if cell:
        freeCellCount, cellType = cell.split(',')
        if freeCellCount:
          currentIndex += int(freeCellCount)+1
        else:
          currentIndex += 1
        row, col = self._linearTo2d(currentIndex)
        if cellType == 'r1' or cellType == 'r3':
          self.rocks.append((row,col))
        elif cellType == 'f1':
          self.goalPositions.append((row,col))
        elif cellType == 's1':
          self.startPositions.append((row,col))
        elif cellType[0:1] == 'c':
          self.checkpoints.append((row, col, len(InternalCellType)-1+int(cellType[1:])))
          self.maxCheckpointCount += 1

  def _get_obs(self):
    mapping = {
      InternalCellType.OPEN.value: CellType.OPEN.value,
      InternalCellType.ROCK.value: CellType.BLOCKED.value,
      InternalCellType.WALL.value: CellType.BLOCKED.value,
      InternalCellType.START.value: CellType.START.value,
      InternalCellType.GOAL.value: CellType.GOAL.value
    }

    def transform(cell):
      if cell >= len(InternalCellType):
        return cell - (len(InternalCellType) - len(CellType))
      return mapping[cell]

    vectorized_transform = np.vectorize(transform)
    transformed_grid = vectorized_transform(self.grid)
    mask = (self.grid == InternalCellType.OPEN.value)
    return { 'board': transformed_grid,
             'action_mask': mask.astype(np.int8) }

  def _get_info(self):
    return {
      'Path length': self.lastPathLength
    }

  def _resetGrid(self):
    # Initialize grid with OPEN cells (which have value 0)
    self.grid = np.zeros(self.gridSize, dtype=np.int32)

  def _randomPos(self):
    row = self.np_random.integers(low=0, high=self.gridSize[0], dtype=np.int32)
    col = self.np_random.integers(low=0, high=self.gridSize[1], dtype=np.int32)
    return (row, col)

  def _generateRandomCheckpoints(self,checkpointCount):
    checkpointVal = len(InternalCellType)
    while checkpointCount>0:
      row, col = self._randomPos()
      pos = (int(row), int(col))

      # Check if the cell is open
      if pos in self.startPositions or pos in self.goalPositions or pos in self.rocks:
        continue
      
      # Place the checkpoint
      self.checkpoints.append((int(row), int(col), checkpointVal))
      checkpointVal += 1
      checkpointCount -= 1

    
  def _generateRandomRocks(self, rocksToPlace:int):
    """Generates a random grid where it is possible to reach the end"""
    while rocksToPlace > 0:
      # Generate a random position
      randomRow, randomCol = self._randomPos()

      # Can only place rocks in open cells
      if self.grid[randomRow][randomCol] != InternalCellType.OPEN.value:
        continue
      
      # Place the rock and test if a path still exists
      self.grid[randomRow][randomCol] = InternalCellType.ROCK.value
      shortestPath = self._calculateShortestPath()
      if shortestPath != 0:
        # Success
        self.rocks.append((int(randomRow),int(randomCol)))
        rocksToPlace -= 1
      else:
        # Failed to place here, reset the cell
        self.grid[randomRow][randomCol] = InternalCellType.OPEN.value

  def _calculateShortestSubpath(self, subStartPos, goalType):
    # Directions for moving: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Create a queue for BFS and add the starting point
    queue = deque([(subStartPos, 0)])
    
    # Set of visited nodes
    visited = set()
    visited.add(subStartPos)
    
    while queue:
      # Get the current position and the path length to it
      (current, pathLength) = queue.popleft()
      
      # If the current position is the goal, return the path
      if self.grid[current[0]][current[1]] == goalType:
        return pathLength
      
      # Explore all the possible directions
      for direction in directions:
        # Calculate the next position
        next_position = (current[0] + direction[0], current[1] + direction[1])
        
        # Check if the next position is within the grid bounds
        if (0 <= next_position[0] < self.gridSize[0]) and (0 <= next_position[1] < self.gridSize[1]):
          # Check if the next position is not an obstacle and not visited
          if self.grid[next_position[0]][next_position[1]] not in [InternalCellType.ROCK.value, InternalCellType.WALL.value] and next_position not in visited:
            # Add the next position to the queue and mark it as visited
            queue.append((next_position, pathLength+1))
            visited.add(next_position)
    
    # There is no path to the goal
    return 0

  def _calculateShortestPathFromSingleStart(self, startPos):
    if len(self.checkpoints) == 0:
      return self._calculateShortestSubpath(startPos, InternalCellType.GOAL.value)

    sum = self._calculateShortestSubpath(startPos, self.checkpoints[0][2])
    if sum == 0:
      # If any path is blocked, the entire path length is 0
      return 0
    for i in range(1, len(self.checkpoints)):
      calculatedPathLength = self._calculateShortestSubpath(self.checkpoints[i-1], self.checkpoints[i][2])
      if calculatedPathLength == 0:
        # If any path is blocked, the entire path length is 0
        return 0
      sum += calculatedPathLength
    calculatedPathLength = self._calculateShortestSubpath(self.checkpoints[-1], InternalCellType.GOAL.value)
    if calculatedPathLength == 0:
      # If any path is blocked, the entire path length is 0
      return 0
    sum += calculatedPathLength
    return sum

  def _calculateShortestPath(self):
    pathLengths = [self._calculateShortestPathFromSingleStart(startPos) for startPos in self.startPositions]
    nonZeroPathLengths = [item for item in pathLengths if item != 0]
    if nonZeroPathLengths:
      return min(nonZeroPathLengths)
    return 0

  def _render_ansi(self):
    ansi_map = {
      InternalCellType.OPEN: '░',  # Open cells
      InternalCellType.ROCK: '█',  # Blocked as a pre-existing part of the map
      InternalCellType.WALL: '#',  # Blocked by player
      InternalCellType.START: 'S', # Start
      InternalCellType.GOAL: 'G'   # Goal
    }
    def getChar(val):
      if val >= len(InternalCellType):
        # Return a character for checkpoints. First checkpoint is A, second is B, etc.
        return chr(ord('A') + val - len(InternalCellType))
      return ansi_map[InternalCellType(val)]

    top_border = "+" + "-" * (self.gridSize[1] * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(getChar(val) for val in row) + '|\n'
    output += top_border + '\n'
    output += f'Remaining walls: {self.remainingWalls}'
    return output
