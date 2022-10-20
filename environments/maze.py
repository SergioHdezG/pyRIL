from gym_miniworld.envs.maze import Maze
from gym_miniworld.params import DEFAULT_PARAMS
from utils.custom_networks import clip
from gym.spaces import Dict, Box
from gym_miniworld.entity import Agent
import numpy as np


class PyMaze(Maze):
    """
    Maze variation for experiments with pyril and clip included inside
    """
    def __init__(self,
                 forward_step=0.7,
                 turn_step=45,
                 num_rows=3,
                 num_cols=3,
                 use_clip=True,
                 domain_rand=False,
                 max_steps=600):

        self.use_clip = use_clip
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        self.observation_space = Box(
            low=0, high=255, shape=(60, 80, 3), dtype=np.uint8
        )

        if self.use_clip:
            obs_dict = {"rgb": self.observation_space}
            self.clipResNet = clip.ResNetCLIPEncoder(Dict(obs_dict), is_habitat=False)

        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=domain_rand
        )

    def step(self, action, resnet=False):
        obs, reward, done, info = super().step(action)

        # reward = self.reward()

        if self.use_clip:
            obs = self.clipResNet.forward(obs)
            obs = obs.squeeze()

        if self.near(self.box):
            # High reward if the agent reaches the goal
            reward += 1
            done = True

        return obs, reward, done, info

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'sky_color',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max('forward_step')

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min([r.min_x for r in self.rooms])
        self.max_x = max([r.max_x for r in self.rooms])
        self.min_z = min([r.min_z for r in self.rooms])
        self.max_z = max([r.max_z for r in self.rooms])

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()
        if self.use_clip:
            obs = self.clipResNet.forward(obs)
            obs = obs.squeeze()
        # Return first observation
        return obs
