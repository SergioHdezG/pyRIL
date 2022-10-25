import os
import shutil
import sys

import habitat
import numpy as np
from gym import spaces
from habitat.core.utils import try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from utils.custom_networks import clip
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict

cv2 = try_cv2_import()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class HM3DRLEnv(habitat.RLEnv):
    """
    Matterport annotated objects: ["wall", "objects", "door", "chair", "window", "ceiling", "picture", "floor", "misc",
    "lighting", "cushion", "table", "cabinet", "curtain", "plant", "shelving", "sink", "mirror", "chest", "towel",
    "stairs", "railing", "column", "counter", "stool", "bed", "sofa", "shower", "appliances", "toilet", "tv",
    "seating", "clothes", "fireplace", "bathtub", "beam", "furniture", "gym equip", "blinds", "board"]

    default configuration for reference: habitat/config/default.py
    """

    def __init__(self, config_paths="configs/RL/objectnav_hm3d_RL.yaml",
                 result_path=os.path.join("development", "images"),
                 task=None,
                 render_on_screen=False,
                 save_video=False,
                 oracle_stop=False,
                 use_clip=False):
        print(f"{bcolors.OKBLUE}Creating new env.{bcolors.ENDC}")

        if not os.path.exists(result_path) and save_video:
            os.makedirs(result_path)
        self.result_path = result_path
        self.config_path = config_paths

        config = habitat.get_config(config_paths=config_paths)
        config.defrost()
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.freeze()

        super().__init__(config=config)
        self.episode_counter = 0
        self.episode_results_path = self.result_path
        self.episode_images = []
        self.save_video = save_video

        self.metadata = {
            'render.modes': ['rgb']
        }
        self.action_space = spaces.Discrete(len(self.config.TASK.POSSIBLE_ACTIONS))
        self.action_list = range(len(self.config.TASK.POSSIBLE_ACTIONS))
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.config.SIMULATOR.RGB_SENSOR.HEIGHT,
                                                   self.config.SIMULATOR.RGB_SENSOR.WIDTH,
                                                   3),
                                            dtype=np.uint8)

        self._task = task
        self._generator = None
        self.first_reset = True
        self.render_on_screen = render_on_screen
        self._previous_measure = None
        self._oracle_stop = oracle_stop
        self.use_clip = use_clip
        if self.use_clip:
            obs_dict = {"rgb": self.observation_space}
            self.clipResNet = clip.ResNetCLIPEncoder(SpaceDict(obs_dict))
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE

    def reset(self):
        self.close()
        if len(self.episode_images) > 0 and self.save_video:
            images_to_video(self.episode_images, self.episode_results_path, "trajectory", verbose=False)

        self.episode_images = []

        observation = super().reset()

        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        self.first_reset = False

        self.episode_counter += 1

        if self.save_video:

            # Define path to save the videos
            self.episode_results_path = os.path.join(
                self.result_path, "shortest_path_example", "%02d" % self.episode_counter
            )

            if os.path.exists(self.episode_results_path):
                shutil.rmtree(self.episode_results_path)
            os.makedirs(self.episode_results_path)
        if self.use_clip:
            observation['rgb'] = self.clipResNet.forward(observation)
        return observation

    def step(self, *args, **kwargs):
        action = self.action_list[args[0]]
        observation, reward, done, info = super().step(action, **kwargs)

        # We save images to create a video later
        im = observation["rgb"]
        top_down_map = self._draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        self.episode_images.append(output_im)
        if self.use_clip:
            observation['rgb'] = self.clipResNet.forward(observation)
        return observation, reward, done, info

    def render(self, mode: str = "rgb"):
        image = super().render(mode=mode)

        if self.render_on_screen:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.episode_results_path, image)
            cv2.waitKey(1)
        return image

    def get_reward_range(self):
        """
        Default habitat implementation from habitat.core.environments.py
        """
        return (
            self.config.TASK.SLACK_REWARD - 1.0,
            self.config.TASK.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        """
        Default habitat implementation from habitat.core.environments.py
        """
        reward = self.config.TASK.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self.config.TASK.SUCCESS_REWARD

        return reward

    def get_done(self, observations):
        """
        Default habitat implementation from habitat.core.environments.py
        """
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        """
        Default habitat implementation from habitat.core.environments.py
        """
        return self.habitat_env.get_metrics()

    def _episode_success(self):
        """
        Determine wether the episode is a successfull one depending on oracle stop or not.
        """
        if self._oracle_stop:
            return self._env.get_metrics()['distance_to_goal'] <= self.config.TASK.SUCCESS.SUCCESS_DISTANCE
        else:
            return self._env.get_metrics()[self._success_measure_name]

    def get_success(self):
        """
        Method to use as a logger for getting success
        Returns: _episode_success
        """
        return self._episode_success()

    def _draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_size
        )

    def close(self):
        cv2.destroyAllWindows()

    # def __getstate__(self):
    #     """See `Object.__getstate__.
    #
    #     Returns:
    #         dict: The instance’s dictionary to be pickled.
    #
    #     """
    #     config_paths = copy.copy(self.config_path)
    #     result_path = copy.copy(self.result_path)
    #     return dict(config_paths=config_paths, result_path=result_path, task=self._task)
    #
    # def __setstate__(self, state):
    #     """See `Object.__setstate__.
    #
    #     Args:
    #         state (dict): Unpickled state of this object.
    #
    #     """
    #     self.__init__(config_paths=state['config_paths'], result_path=state['result_path'], task=state['task'])


class HM3DRLVecEnv(habitat.VectorEnv):
    """
    Matterport annotated objects: ["wall", "objects", "door", "chair", "window", "ceiling", "picture", "floor", "misc",
    "lighting", "cushion", "table", "cabinet", "curtain", "plant", "shelving", "sink", "mirror", "chest", "towel",
    "stairs", "railing", "column", "counter", "stool", "bed", "sofa", "shower", "appliances", "toilet", "tv",
    "seating", "clothes", "fireplace", "bathtub", "beam", "furniture", "gym equip", "blinds", "board"]

    default configuration for reference: habitat/config/default.py
    """

    def __init__(self,
                 make_env_fn,
                 env_fn_args=None,
                 auto_reset_done: bool = True,
                 multiprocessing_start_method: str = "forkserver",
                 workers_ignore_signals: bool = False):

        super().__init__(make_env_fn=make_env_fn,
                         env_fn_args=env_fn_args,
                         auto_reset_done=auto_reset_done,
                         multiprocessing_start_method=multiprocessing_start_method,
                         workers_ignore_signals=workers_ignore_signals)
