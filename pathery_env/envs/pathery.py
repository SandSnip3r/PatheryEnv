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
  ICE = 5
# Checkpoints follow the last item

class ObservationCellType(Enum):
  OPEN = 0
  BLOCKED = 1
  START = 2
  GOAL = 3
  ICE = 4
# Checkpoints follow the last item

def createRandomNormal(render_mode, **kwargs):
  return PatheryEnv.randomNormal(render_mode, **kwargs)

def fromMapString(render_mode, map_string, **kwargs):
  return PatheryEnv.fromMapString(render_mode, map_string, **kwargs)

class PatheryEnv(gym.Env):
  OBSERVATION_BOARD_STR = 'board'
  metadata = {"render_modes": ["ansi"], "render_fps": 4}

  @classmethod
  def randomNormal(cls, render_mode, **kwargs):
    return cls(render_mode=render_mode, **kwargs)

  @classmethod
  def fromMapString(cls, render_mode, map_string, **kwargs):
    return cls(render_mode=render_mode, map_string=map_string, **kwargs)

  def __init__(self, render_mode, map_string=None):
    self.randomMap = (map_string == None)

    self.startPositions = []
    self.goalPositions = []
    self.rocks = []
    self.ice = []
    self.checkpoints = []

    if map_string is not None:
      self._initializeFromMapString(map_string)
    else:
      # Size and wall count are hard coded for random maps
      self.gridSize = (9, 17)
      self.wallsToPlace = 14
      self.maxCheckpointCount = 2

    # Observation space: Each cell type is a discrete value, checkpoints are dynamically added on the end
    self.observation_space = spaces.Dict()
    self.observation_space[PatheryEnv.OBSERVATION_BOARD_STR] = spaces.MultiDiscrete(np.full((self.gridSize[0], self.gridSize[1]), len(ObservationCellType) + self.maxCheckpointCount))

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

    if self.randomMap:
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

    # Place rocks
    for rockPos in self.rocks:
      self.grid[rockPos[0]][rockPos[1]] = InternalCellType.ROCK.value

    # Place ice
    for icePos in self.ice:
      self.grid[icePos[0]][icePos[1]] = InternalCellType.ICE.value

    # Place checkpoints
    for row, col, checkpointIndex in self.checkpoints:
      self.grid[row][col] = self._checkpointIndexToCellValue(checkpointIndex)

    # Save checkpoint indices (rather than needing to repeatedly dedup them on every pathfind)
    self.checkpointIndices = sorted(list({self._checkpointIndexToCellValue(index) for _,_,index in self.checkpoints}))

    # Finally, random rock placement must be done after everything else has been placed so that we can check that no rock blocks any path
    if self.randomMap:
      # Pick rocks
      # This also sets self.lastPath
      self._generateRandomRocks(rocksToPlace=14)
    else:
      self.lastPath = self._calculateShortestPath()

    # Keep track of path length
    self.lastPathLength = len(self.lastPath)

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def step(self, action):
    tupledAction = (action[0], action[1])
    if self.grid[tupledAction[0]][tupledAction[1]] != InternalCellType.OPEN.value:
      # Invalid position; reward is -1, episode terminates
      return self._get_obs(), -1, True, False, self._get_info()

    self.grid[tupledAction[0]][tupledAction[1]] = InternalCellType.WALL.value
    self.remainingWalls -= 1
    terminated = self.remainingWalls == 0

    if tupledAction in self.lastPath:
      # Only repath if the placed wall is on the current shortest path
      self.lastPath = self._calculateShortestPath()
      pathLength = len(self.lastPath)

      if pathLength == 0:
        # Blocks path; reward is -1, episode terminates
        self.lastPathLength = 0
        return self._get_obs(), -1, True, False, self._get_info()

      reward = pathLength - self.lastPathLength
      self.rewardSoFar += reward
      if reward < 0:
        raise ValueError(f'Reward is negative: {reward}')
      self.lastPathLength = pathLength
    else:
      reward = 0

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
    # https://www.pathery.com/mapeditor
    # Map string format
    # <width;int>.<height;int>.<num walls;int>.<name;string>...<unk>:([<number of open cells>],<cell type>.)*
    # Cell types:
    #   r1: Map-pre-placed rock
    #   r2: Player-placed wall
    #   r3: Map boundary rock
    #   z5: Ice
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
        elif cellType == 'z5':
          self.ice.append((row,col))
        elif cellType == 'f1':
          self.goalPositions.append((row,col))
        elif cellType == 's1':
          self.startPositions.append((row,col))
        elif cellType[0:1] == 'c':
          # Add checkpoints to a list so that we can later sort them by index. This lets us receive them out of order.
          self.checkpoints.append((row, col, int(cellType[1:])-1))
        else:
          print(f'WARNING: When parsing map string, encountered unknown cell "{cellType}" at pos ({row},{col}).')
    # Count the number of unique checkpoint indices.
    self.maxCheckpointCount = len({x[2] for x in self.checkpoints})

  def _get_obs(self):
    mapping = {
      InternalCellType.OPEN.value: ObservationCellType.OPEN.value,
      InternalCellType.ROCK.value: ObservationCellType.BLOCKED.value,
      InternalCellType.WALL.value: ObservationCellType.BLOCKED.value,
      InternalCellType.START.value: ObservationCellType.START.value,
      InternalCellType.GOAL.value: ObservationCellType.GOAL.value,
      InternalCellType.ICE.value: ObservationCellType.ICE.value
    }

    def transform(cell):
      if cell >= len(InternalCellType):
        return cell - (len(InternalCellType) - len(ObservationCellType))
      return mapping[cell]

    vectorized_transform = np.vectorize(transform)
    return {
      PatheryEnv.OBSERVATION_BOARD_STR: vectorized_transform(self.grid)
    }

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
      if pos in self.startPositions or pos in self.goalPositions or pos in self.rocks or any(pos == t[:len(pos)] for t in self.checkpoints):
        continue
      
      # Place the checkpoint
      self.checkpoints.append((int(row), int(col), checkpointVal))
      checkpointVal += 1
      checkpointCount -= 1

    
  def _generateRandomRocks(self, rocksToPlace:int):
    """Generates a random grid where it is possible to reach the end"""
    self.lastPath = self._calculateShortestPath()
    while rocksToPlace > 0:
      # Generate a random position
      randomRow, randomCol = self._randomPos()

      # Can only place rocks in open cells
      if self.grid[randomRow][randomCol] != InternalCellType.OPEN.value:
        continue
      
      # Place the rock and test if a path still exists
      self.grid[randomRow][randomCol] = InternalCellType.ROCK.value
      needToRePath = len(self.lastPath) == 0 or (randomRow, randomCol) in self.lastPath
      if needToRePath:
        self.lastPath = self._calculateShortestPath()
      shortestPathLength = len(self.lastPath)
      if shortestPathLength != 0:
        # Success
        self.rocks.append((int(randomRow),int(randomCol)))
        rocksToPlace -= 1
      else:
        # Failed to place here, reset the cell
        self.grid[randomRow][randomCol] = InternalCellType.OPEN.value

  def _calculateShortestSubpath(self, subStartPos, goalType):
    # Directions for moving: up, right, down, left (this is the order preferred by Pathery)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Create a queue for BFS and add the starting point
    start = (subStartPos, None)
    queue = deque([start])
    
    # Set of visited nodes
    visited = set()
    visited.add(start)
    prev = {}

    def buildPath(end):
      path = []
      while end in prev:
        path.append(end[0])
        end = prev[end]
      return path[::-1]
    
    while queue:
      current = queue.popleft()
      currentPosition, currentDirection = current
      
      # If the current position is the goal, return the path
      if self.grid[currentPosition[0]][currentPosition[1]] == goalType:
        return buildPath(current)
      
      # Explore all the possible directions
      for direction in (directions if currentDirection is None else [currentDirection]):
        # Calculate the next position
        nextPosition = (currentPosition[0] + direction[0], currentPosition[1] + direction[1])
        
        # Check if the next position is within the grid bounds
        if (0 <= nextPosition[0] < self.gridSize[0]) and (0 <= nextPosition[1] < self.gridSize[1]):
          # Check if the next position is not an obstacle and not visited
          if self.grid[nextPosition[0]][nextPosition[1]] not in [InternalCellType.ROCK.value, InternalCellType.WALL.value]:
            next = (nextPosition, (direction if self.grid[nextPosition[0]][nextPosition[1]] == InternalCellType.ICE.value else None))
            if next not in visited:
              # Add the next position to the queue and mark it as visited
              queue.append(next)
              visited.add(next)
              prev[next] = current
    
    # There is no path to the goal
    return []

  def _calculateShortestPath(self):
    if len(self.startPositions) > 1:
      raise ValueError('Do not support multiple starts')
    startPos = self.startPositions[0]

    if len(self.checkpointIndices) == 0:
      # No checkpoints, path directly from the start to the goal.
      return self._calculateShortestSubpath(startPos, InternalCellType.GOAL.value)

    overallPath = self._calculateShortestSubpath(startPos, self.checkpointIndices[0])
    if len(overallPath) == 0:
      # If any sub-path is blocked, the entire path is blocked
      return []
    for checkpointIndex in self.checkpointIndices[1:]:
      subPath = self._calculateShortestSubpath(overallPath[-1], checkpointIndex)
      if len(subPath) == 0:
        # If any sub-path is blocked, the entire path is blocked
        return []
      overallPath.extend(subPath)
    finalSubPath = self._calculateShortestSubpath(overallPath[-1], InternalCellType.GOAL.value)
    if len(finalSubPath) == 0:
      # If any sub-path is blocked, the entire path is blocked
      return []
    overallPath.extend(finalSubPath)
    return overallPath

  def _render_ansi(self):
    ansi_map = {
      InternalCellType.OPEN: ' ',  # Open cells
      InternalCellType.ROCK: '█',  # Blocked as a pre-existing part of the map
      InternalCellType.WALL: '#',  # Blocked by player
      InternalCellType.START: 'S', # Start
      InternalCellType.GOAL: 'G',   # Goal
      InternalCellType.ICE: '░'  # Ice cells
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

  def _checkpointIndexToCellValue(self, checkpointIndex):
    return len(InternalCellType) + checkpointIndex