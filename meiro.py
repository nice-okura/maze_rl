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

    # fieldmap
    WAY = 0
    PLAYER = 1
    GOAL = 5
    WALL = 9

    def reset_fieldmap(self):
        self.fieldmap = np.zeros((self.grid_size, self.grid_size))
        self.fieldmap[3, 5] = self.WALL
        self.fieldmap[3, 4] = self.WALL
        self.fieldmap[3, 3] = self.WALL
        self.fieldmap[3, 2] = self.WALL
        self.fieldmap[3, 1] = self.WALL
        self.fieldmap[2, 1] = self.WALL
        self.fieldmap[1, 1] = self.WALL
        self.fieldmap[0, 1] = self.WALL
        self.fieldmap[4, 5] = self.WALL
        self.fieldmap[5, 5] = self.WALL
        self.fieldmap[5, 4] = self.WALL
        self.fieldmap[5, 3] = self.WALL
        self.fieldmap[5, 2] = self.WALL
        self.fieldmap[5, 1] = self.WALL
        self.fieldmap[5, 0] = self.WALL

        # PLAYER
        self.fieldmap[int(self.grid_size * 0.5), int(self.grid_size * 0.5)] = self.PLAYER
        self.agent_pos = np.array([int(self.grid_size * 0.5), int(self.grid_size * 0.5)])

        # GOAL
        self.fieldmap[0, 0] = self.GOAL
        self.goal_pos = np.array([0, 0])

    def __init__(self, grid_size=5):
        super(meiro, self).__init__()

        # fields
        self.grid_size = grid_size
        self.reset_fieldmap()
        self.action = 0
        self.reward = 0

        # action num
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)

        # low=input_lower_limit, high=input_higher_limit
        self.observation_space = spaces.Box(
            low=self.WAY, high=self.WALL, shape=(grid_size, grid_size,), dtype=np.float32)

    def reset(self):
        # initial position
        self.reset_fieldmap()
        # numpy only
        return np.array(self.fieldmap).astype(np.float32)

    def step(self, action):
        prev_pos = self.agent_pos.copy()
        collision_wall = False

        # fetch action
        if self.A_L == action:
            self.agent_pos[0] -= 1
        if self.A_R == action:
            self.agent_pos[0] += 1
        if self.A_U == action:
            self.agent_pos[1] -= 1
        if self.A_D == action:
            self.agent_pos[1] += 1

        if not (0 < self.agent_pos[0] < self.grid_size) or not (0 < self.agent_pos[1] < self.grid_size):
            collision_wall = True

        # moving limit
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size - 1)

        # wall check
        if self.fieldmap[self.agent_pos[0], self.agent_pos[1]] == self.WALL:
            collision_wall = True
            # collision wall and set previous pos
            self.agent_pos = prev_pos.copy()

        # update fieldmap
        # print(self.agent_pos, prev_pos)
        self.fieldmap[prev_pos[0], prev_pos[1]] = self.WAY
        self.fieldmap[self.agent_pos[0], self.agent_pos[1]] = self.PLAYER

        # goal and reward
        done = False

        # reward = 0
        reward = (self.grid_size * 0.5) - np.abs(self.goal_pos[0] - self.agent_pos[0])
        reward += (self.grid_size * 0.5) - np.abs(self.goal_pos[1] - self.agent_pos[1])
        # reward *= 0.05

        if collision_wall:
            # print("COLLLLLLLLLLLLLLLLL")
            done = True
            reward -= 10.0

        if self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1]:
            done = True
            reward += 1000

        # fetch variables
        self.action = action
        self.reward = reward

        # infomation
        info = {}
        return np.array(self.fieldmap).astype(np.float32), reward, done, info

    # draw
    def render(self, mode='human', close=False):
        if mode != 'human':
            raise NotImplementedError()
        draw_map = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.fieldmap[j, i] == self.GOAL:
                    draw_map += "G "
                else:
                    if self.fieldmap[j, i] == self.PLAYER:
                        draw_map += "O "
                    elif self.fieldmap[j, i] == self.WALL:
                        draw_map += "X "
                    else:
                        draw_map += ". "
            draw_map += '\n'
        print("Action:", self.action, "Agent:",
              self.agent_pos, "Reward:", self.reward)
        print(draw_map)
