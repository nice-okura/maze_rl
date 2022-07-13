import numpy as np
import gym
from gym import spaces


class meiro(gym.Env):
  metadata = {'render.modes': ['human']}

  # Actions
  A_L = 0
  A_R = 1
  A_U = 2
  A_D = 3

  # wall
  WALL = 9

  def __init__(self, grid_size=5):
    super(meiro, self).__init__()
    # fields
    self.grid_size = grid_size
    self.fieldmap = np.zeros((grid_size, grid_size))
    self.fieldmap[1, 0] = self.WALL
    self.fieldmap[2, 1] = self.WALL
    self.fieldmap[4, 2] = self.WALL
    self.fieldmap[6, 3] = self.WALL
    self.fieldmap[7, 4] = self.WALL
    self.fieldmap[2, 5] = self.WALL
    self.fieldmap[1, 6] = self.WALL
    self.agent_pos = np.array([int(grid_size * 0.5), int(grid_size * 0.5)])
    self.goal_pos = np.array([0, 0])
    self.action = 0
    self.reward = 0
    # action num
    n_actions = 4
    self.action_space = spaces.Discrete(n_actions)
    # low=input_lower_limit, high=input_higher_limit
    self.observation_space = spaces.Box(
        low=0, high=self.grid_size, shape=(2,), dtype=np.float32)

  def reset(self):
    # initial position
    self.agent_pos = np.array(
        [int(self.grid_size * 0.5), int(self.grid_size * 0.5)])
    # numpy only
    return np.array(self.agent_pos).astype(np.float32)

  def step(self, action):
    prev_pos = self.agent_pos.copy()

    # fetch action
    if self.A_L == action:
      self.agent_pos[0] -= 1
    if self.A_R == action:
      self.agent_pos[0] += 1
    if self.A_U == action:
      self.agent_pos[1] -= 1
    if self.A_D == action:
      self.agent_pos[1] += 1

    # moving limit
    self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size - 1)
    self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size - 1)

    # wall check
    collision_wall = True if self.fieldmap[self.agent_pos[0], self.agent_pos[1]] == self.WALL else False
    if collision_wall:
      # collision wall and set previous pos
      self.agent_pos = prev_pos.copy()

    # goal and reward
    done = False
    # reward = 0
    reward = (self.grid_size * 0.5) - \
        np.abs(self.goal_pos[0] - self.agent_pos[0])
    reward += (self.grid_size * 0.5) - \
        np.abs(self.goal_pos[1] - self.agent_pos[1])
    reward *= 0.05

    if collision_wall:
      print("COLLLLLLLLLLLLLLLLL")
      done = True
      reward -= 100.0

    if self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1]:
      done = True
      reward += 1000

    # fetch variables
    self.action = action
    self.reward = reward

    # infomation
    info = {'fieldmap': self.fieldmap}
    return np.array(self.agent_pos).astype(np.float32), reward, done, info

  # draw
  def render(self, mode='human', close=False):
    if mode != 'human':
      raise NotImplementedError()
    draw_map = ""
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        if self.goal_pos[0] == j and self.goal_pos[1] == i:
          draw_map += "G "
        else:
          if self.agent_pos[0] == j and self.agent_pos[1] == i:
            draw_map += "O "
          elif self.fieldmap[j, i] == self.WALL:
            draw_map += "X "
          else:
            draw_map += ". "
      draw_map += '\n'
    print("Action:", self.action, "Agent:",
          self.agent_pos, "Reward:", self.reward)
    print(draw_map)
