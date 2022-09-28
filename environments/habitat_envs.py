import copy
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


class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config_paths="configs/tasks/pointnav.yaml", result_path=os.path.join("development", "images"),
                 task=None):

        if not os.path.exists(result_path):
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

        self.metadata = {
            'render.modes': ['rgb']
        }
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                                               spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)))

        self._task = task
        self._generator = None

    def reset(self):
        # Al final de cada episodio guardar un video del recorido del robot
        if len(self.episode_images) > 0:
            images_to_video(self.episode_images, self.episode_results_path, "trajectory", verbose=False)

        self.episode_images = []
        observation = super().reset()

        self.episode_counter += 1

        # Definir ruta paga guardar las ejecuciones
        self.episode_results_path = os.path.join(
            self.result_path, "shortest_path_example", "%02d" % self.episode_counter
        )

        if os.path.exists(self.episode_results_path):
            shutil.rmtree(self.episode_results_path)
        os.makedirs(self.episode_results_path)

        return [observation['rgb'], observation['pointgoal_with_gps_compass']]

    def step(self, *args, **kwargs):
        observation, reward, done, info = super().step(*args, **kwargs)

        # Guardar video de las ejecuciones en un archivo
        im = observation["rgb"]
        top_down_map = self._draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        self.episode_images.append(output_im)

        reward = self._get_reward(info['distance_to_goal'])
        return [observation['rgb'], observation['pointgoal_with_gps_compass']], reward, done, info

    def render(self, mode: str = "rgb", print_on_screen=False):
        image = super().render(mode=mode)

        if print_on_screen:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.episode_results_path, image)
            cv2.waitKey(1)
        return image

    def get_reward_range(self):
        return [-20, 0]

    # Obligatorio: solo recibe la observación, por lo que es muy dificil proponer una recompensa
    def get_reward(self, observations):
        return 0

    # Calcula la recompensa real en función a la distancia con el punto objetivo
    def _get_reward(self, distance2goal):
        return - distance2goal

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def _draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_size
        )

    def sample_tasks(self, num_tasks):
        generators = [np.random.random((2,)) for _ in range(num_tasks)]
        tasks = [{'generator': generator} for generator in generators]
        return tasks

    def set_task(self, task):
        self._task = task
        self._generator = task['generator']

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        config_paths = copy.copy(self.config_path)
        result_path = copy.copy(self.result_path)
        return dict(config_paths=config_paths, result_path=result_path, task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(config_paths=state['config_paths'], result_path=state['result_path'], task=state['task'])


class SimpleRLEnvRGB(SimpleRLEnv):
    def __init__(self, config_paths="configs/tasks/pointnav.yaml", result_path=os.path.join("development", "images"),
                 task=None):
        super().__init__(config_paths=config_paths, result_path=result_path)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )

    def reset(self):
        observation = super().reset()
        im = observation[0]
        return im

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        im = observations[0]

        return im, reward, done, info


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
                 save_video=False):
        print(f"{bcolors.OKBLUE}Creando un nuevo entorno.{bcolors.ENDC}")

        if not os.path.exists(result_path):
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
        self.action_space = spaces.Discrete(6)  # FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN, STOP
        self.action_list = [HabitatSimActions.MOVE_FORWARD,
                            HabitatSimActions.TURN_LEFT,
                            HabitatSimActions.TURN_RIGHT,
                            HabitatSimActions.LOOK_UP,
                            HabitatSimActions.LOOK_DOWN,
                            HabitatSimActions.STOP]

        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        #                                        spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)))
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        self._task = task
        self._generator = None
        self.first_reset = True
        self.render_on_screen = render_on_screen

    def reset(self):
        # print(f"{bcolors.OKCYAN} Reseteando el entorno.{bcolors.ENDC}")

        self.close()
        if len(self.episode_images) > 0 and self.save_video:
            images_to_video(self.episode_images, self.episode_results_path, "trajectory", verbose=False)

        self.episode_images = []

        sys.stdout = open(os.devnull, 'w')
        # if self.first_reset:
        observation = super().reset()
        sys.stdout = sys.__stdout__

        self.first_reset = False
        # else:
        #     observation = super().step(HabitatSimActions.STOP)
        self.episode_counter += 1

        if self.save_video:

            # Definir ruta paga guardar las ejecuciones
            self.episode_results_path = os.path.join(
                self.result_path, "shortest_path_example", "%02d" % self.episode_counter
            )

            if os.path.exists(self.episode_results_path):
                shutil.rmtree(self.episode_results_path)
            os.makedirs(self.episode_results_path)

        return observation['rgb']  # observation['pointgoal_with_gps_compass']] # cuidado con el formato de la
        # observación porque si no esta bien te hace un flatten en el get_action() de garage/src/garage/torch/policies/
        # sthocastic_policy.py

    def step(self, *args, **kwargs):
        action = self.action_list[args[0]]
        observation, reward, done, info = super().step(action, **kwargs)

        if action == HabitatSimActions.STOP:
            done = True

        # Guardar video de las ejecuciones en un archivo
        im = observation["rgb"]
        top_down_map = self._draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        self.episode_images.append(output_im)

        reward = self._get_reward(info['distance_to_goal'])
        return observation[
                   'rgb'], reward, done, info  # , observation['pointgoal_with_gps_compass']], reward, done, info
        # cuidado con el formato de la
        # observación porque si no esta bien te hace un flatten en el get_action() de garage/src/garage/torch/policies/
        # sthocastic_policy.py

    def render(self, mode: str = "rgb"):
        image = super().render(mode=mode)

        if self.render_on_screen:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.episode_results_path, image)
            cv2.waitKey(1)
        return image

    def get_reward_range(self):
        return [-20, 0]

    # Obligatorio: solo recibe la observación, por lo que es muy dificil proponer una recompensa
    def get_reward(self, observations):
        return 0

    # Calcula la recompensa real en función a la distancia con el punto objetivo
    def _get_reward(self, distance2goal):
        return - distance2goal

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def _draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_size
        )

    def sample_tasks(self, num_tasks):
        generators = [np.random.random((2,)) for _ in range(num_tasks)]
        tasks = [{'generator': generator} for generator in generators]
        return tasks

    def set_task(self, task):
        self._task = task
        self._generator = task['generator']

    def close(self):
        cv2.destroyAllWindows()

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        config_paths = copy.copy(self.config_path)
        result_path = copy.copy(self.result_path)
        return dict(config_paths=config_paths, result_path=result_path, task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(config_paths=state['config_paths'], result_path=state['result_path'], task=state['task'])


class HM3DMetaRLEnv(habitat.RLEnv):
    """
    En cada reset se crea una escena nueva y un objetivo nuevo. Falta que esto se realize en la asignación de tarea.
    """

    def __init__(self, config_paths="configs/tasks/objectnav_hm3d.yaml",
                 result_path=os.path.join("development", "images"),
                 task=None):
        print(f"{bcolors.OKBLUE}Creando un nuevo entorno.{bcolors.ENDC}")

        if not os.path.exists(result_path):
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

        self.metadata = {
            'render.modes': ['rgb']
        }
        self.action_space = spaces.Discrete(6)  # FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN, STOP
        self.action_list = [HabitatSimActions.MOVE_FORWARD,
                            HabitatSimActions.TURN_LEFT,
                            HabitatSimActions.TURN_RIGHT,
                            HabitatSimActions.LOOK_UP,
                            HabitatSimActions.LOOK_DOWN,
                            HabitatSimActions.STOP]

        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        #                                        spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)))
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        self._task = task
        self._generator = None

    def reset(self):
        print(f"{bcolors.OKCYAN} Reseteando el entorno.{bcolors.ENDC}")

        # Al final de cada episodio guardar un video del recorido del robot
        if len(self.episode_images) > 0:
            images_to_video(self.episode_images, self.episode_results_path, "trajectory")

        self.episode_images = []
        observation = super().reset()

        self.episode_counter += 1

        # Definir ruta paga guardar las ejecuciones
        self.episode_results_path = os.path.join(
            self.result_path, "shortest_path_example", "%02d" % self.episode_counter
        )

        if os.path.exists(self.episode_results_path):
            shutil.rmtree(self.episode_results_path)
        os.makedirs(self.episode_results_path)

        return observation['rgb']  # observation['pointgoal_with_gps_compass']] # cuidado con el formato de la
        # observación porque si no esta bien te hace un flatten en el get_action() de garage/src/garage/torch/policies/
        # sthocastic_policy.py

    def step(self, *args, **kwargs):
        observation, reward, done, info = super().step(*args, **kwargs)

        # Guardar video de las ejecuciones en un archivo
        im = observation["rgb"]
        top_down_map = self._draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        self.episode_images.append(output_im)

        reward = self._get_reward(info['distance_to_goal'])
        return observation[
                   'rgb'], reward, done, info  # , observation['pointgoal_with_gps_compass']], reward, done, info
        # cuidado con el formato de la
        # observación porque si no esta bien te hace un flatten en el get_action() de garage/src/garage/torch/policies/
        # sthocastic_policy.py

    def render(self, mode: str = "rgb", print_on_screen=False):
        image = super().render(mode=mode)

        if print_on_screen:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.episode_results_path, image)
            cv2.waitKey(1)
        return image

    def get_reward_range(self):
        return [-20, 0]

    # Obligatorio: solo recibe la observación, por lo que es muy dificil proponer una recompensa
    def get_reward(self, observations):
        return 0

    # Calcula la recompensa real en función a la distancia con el punto objetivo
    def _get_reward(self, distance2goal):
        return - distance2goal

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def _draw_top_down_map(self, info, output_size):
        return maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_size
        )

    def sample_tasks(self, num_tasks):
        generators = [np.random.random((2,)) for _ in range(num_tasks)]
        tasks = [{'generator': generator} for generator in generators]
        return tasks

    def set_task(self, task):
        self._task = task
        self._generator = task['generator']

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        config_paths = copy.copy(self.config_path)
        result_path = copy.copy(self.result_path)
        return dict(config_paths=config_paths, result_path=result_path, task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(config_paths=state['config_paths'], result_path=state['result_path'], task=state['task'])
