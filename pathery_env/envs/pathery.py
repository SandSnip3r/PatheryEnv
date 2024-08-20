from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
  right = 0
  up = 1
  left = 2
  down = 3

class CellType(Enum):
  OPEN = 0
  BLOCKED_PRE_EXISTING = 1
  BLOCKED_PLAYER_PLACED = 2
  START = 3
  GOAL = 4

class PatheryEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

  def __init__(self, render_mode=None):
    self.size = 4  # The size of the square grid
    self.grid_size = (self.size, self.size)
    self.grid = np.zeros(self.grid_size, dtype=np.int32)  # Initialize grid with OPEN cells

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.MultiDiscrete(np.full((self.grid_size[0], self.grid_size[1]), len(CellType)))

    self.window_size = 512  # The size of the PyGame window

    # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
    self.action_space = spaces.Discrete(4)

    """
    The following dictionary maps abstract actions from `self.action_space` to 
    the direction we will walk in if that action is taken.
    i.e. 0 corresponds to "right", 1 to "up" etc.
    """
    self._action_to_direction = {
      Actions.right.value: np.array([1, 0]),
      Actions.up.value: np.array([0, 1]),
      Actions.left.value: np.array([-1, 0]),
      Actions.down.value: np.array([0, -1]),
    }

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    """
    If human-rendering is used, `self.window` will be a reference
    to the window that we draw to. `self.clock` will be a clock that is used
    to ensure that the environment is rendered at the correct framerate in
    human-mode. They will remain `None` until human-mode is used for the
    first time.
    """
    self.window = None
    self.clock = None

  def _get_obs(self):
    return self.grid

  def _get_info(self):
    return {
      "distance": np.linalg.norm(
        self._agent_location - self._target_location, ord=1
      )
    }

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Start in top left
    self.grid[0][0] = CellType.START.value
    # End in bottom right
    self.grid[self.grid_size[0]-1][self.grid_size[1]-1] = CellType.GOAL.value

    # Choose the agent's location uniformly at random
    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    # We will sample the target's location randomly until it does not
    # coincide with the agent's location
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
      self._target_location = self.np_random.integers(
        0, self.size, size=2, dtype=int
      )

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info

  def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
    direction = self._action_to_direction[action]
    # We use `np.clip` to make sure we don't leave the grid
    self._agent_location = np.clip(
      self._agent_location + direction, 0, self.size - 1
    )
    # An episode is done iff the agent has reached the target
    terminated = np.array_equal(self._agent_location, self._target_location)
    reward = 1 if terminated else 0  # Binary sparse rewards
    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()
    elif self.render_mode == "ansi":
      return self._render_ansi()

  def _render_ansi(self):
    ansi_map = {
      CellType.OPEN: '░',  # Open cells
      CellType.BLOCKED_PRE_EXISTING: '█',  # Blocked by pre-existing
      CellType.BLOCKED_PLAYER_PLACED: '#',  # Blocked by player
      CellType.START: 'S',  # Start
      CellType.GOAL: 'G'   # Goal
    }
    top_border = "+" + "-" * (self.size * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(ansi_map[CellType(val)] for val in row) + '|\n'
    output += top_border
    return output
    
  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode((self.window_size, self.window_size))
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = (
      self.window_size / self.size
    )  # The size of a single grid square in pixels

    # First we draw the target
    pygame.draw.rect(
      canvas,
      (255, 0, 0),
      pygame.Rect(
        pix_square_size * self._target_location,
        (pix_square_size, pix_square_size),
      ),
    )
    # Now we draw the agent
    pygame.draw.circle(
      canvas,
      (0, 0, 255),
      (self._agent_location + 0.5) * pix_square_size,
      pix_square_size / 3,
    )

    # Finally, add some gridlines
    for x in range(self.size + 1):
      pygame.draw.line(
        canvas,
        0,
        (0, pix_square_size * x),
        (self.window_size, pix_square_size * x),
        width=3,
      )
      pygame.draw.line(
        canvas,
        0,
        (pix_square_size * x, 0),
        (pix_square_size * x, self.window_size),
        width=3,
      )

    if self.render_mode == "human":
      # The following line copies our drawings from `canvas` to the visible window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # We need to ensure that human-rendering occurs at the predefined framerate.
      # The following line will automatically add a delay to
      # keep the framerate stable.
      self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
      )

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()
