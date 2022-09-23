import os
import shutil
import habitat
import copy
from habitat import RLEnv
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
from gym import spaces
from habitat.core.utils import try_cv2_import

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

    # def __deepcopy__(self, memodict={}):
    #     config_paths = copy.copy(self.config_path)
    #     result_path = copy.copy(self.result_path)
    #     copied_env = type(self)(config_paths=config_paths, result_path=result_path)
    #     # memodict[id(self)] = copied_env
    #     # copy._member1 = self._member1
    #     # copy._member2 = deepcopy(self._member2, memo)
    #
    #     # TODO [sergio]: Esta copied_env._np_random es supceptible de ser cambiada o modificada entre tareas
    #     copied_env._np_random = copy.deepcopy(self._np_random)
    #     copied_env._core_env_config = copy.deepcopy(copied_env._core_env_config)
    #     copied_env = deepcopy_habitat_RLEnv(copied_env, self)
    #
    #     copied_env.action_space = copy.deepcopy(self.action_space)
    #     # TODO [sergio]: Esta copied_env.config es supceptible de ser cambiada o modificada entre tareas
    #     # copied_env.config = copy.copy(self.config)  # copy # no puedo copiarlo
    #     copied_env.config_path = copy.deepcopy(self.config_path)
    #     # copied_env.current_episode = copy.copy(self.current_episode)  # copy # no puedo copiarlo
    #     copied_env.episode_counter = copy.deepcopy(self.episode_counter)
    #     copied_env.episode_images = copy.deepcopy(self.episode_images)
    #     # TODO [sergio]: Esta copied_env.episode_results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.episode_results_path = copy.deepcopy(self.episode_results_path)  # copy # no puedo copiarlo
    #     copied_env.episodes = copy.deepcopy(self.episodes)
    #     # copied_env.habitat_env = copy.copy(self.habitat_env)  # copy # no puedo copiarlo
    #     copied_env.metadata = copy.deepcopy(self.metadata)
    #     copied_env.number_of_episodes = copy.deepcopy(self.number_of_episodes)
    #     copied_env.observation_space = copy.deepcopy(self.observation_space)
    #     # TODO [sergio]: Esta copied_env.results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.result_path = copy.deepcopy(self.result_path)
    #     copied_env.reward_range = copy.deepcopy(self.reward_range)
    #     copied_env.spec = copy.deepcopy(self.spec)
    #
    #     return copied_env


def deepcopy_habitat_RLEnv(new_env, old_env):
    # new_env._env = copy.deepcopy(old_env._env)
    new_env._env._current_episode = copy.deepcopy(old_env._env._current_episode)
    new_env._env._episode_iterator = copy.deepcopy(old_env._env._episode_iterator)
    new_env._env._episode_over = copy.deepcopy(old_env._env._episode_over)
    new_env._env._episode_start_time = copy.deepcopy(old_env._env._episode_start_time)
    new_env._env._config = copy.deepcopy(old_env._env._config)
    new_env._env._dataset = copy.deepcopy(old_env._env._dataset)
    new_env._env._elapsed_steps = copy.deepcopy(old_env._env._elapsed_steps)
    new_env._env._episode_force_changed = copy.deepcopy(old_env._env._episode_force_changed)
    new_env._env._max_episode_seconds = copy.deepcopy(old_env._env._max_episode_seconds)
    new_env._env._max_episode_steps = copy.deepcopy(old_env._env._max_episode_steps)

    new_env._env.number_of_episodes = copy.deepcopy(old_env._env.number_of_episodes)
    new_env._env.action_space = copy.deepcopy(old_env._env.action_space)
    new_env._env.observation_space = copy.deepcopy(old_env._env.observation_space)

    new_env = deepcopy_sim(new_env, old_env)
    new_env = deepcopy_task(new_env, old_env)
    return new_env


def deepcopy_task(new_env, old_env):
    # new_env._env.task.action_space = copy.copy(old_env._env.task.action_space)  # copy  # no puedo copiarlo
    new_env._env.task.actions = copy.copy(old_env._env.task.actions)  # copy
    new_env._env.task.measurements = copy.copy(old_env._env.task.measurements)  # copy
    new_env._env.task.sensor_suite = copy.copy(old_env._env.task.sensor_suite)  # copy
    new_env._env.task._action_keys = copy.deepcopy(old_env._env.task._action_keys)
    new_env._env.task._config = copy.deepcopy(old_env._env.task._config)
    new_env._env.task._dataset = copy.deepcopy(old_env._env.task._dataset)
    new_env._env.task._is_episode_active = copy.deepcopy(old_env._env.task._is_episode_active)
    # TODO [sergio]: comprobar que esta asignación no produce resultados extraños
    new_env._env.task._sim = old_env._env.sim
    return new_env


def deepcopy_sim(new_env, old_env):
    new_env._env.sim._Simulator__last_state = copy.deepcopy(old_env._env.sim._Simulator__last_state)
    new_env._env.sim._Simulator__sensors = copy.copy(old_env._env.sim._Simulator__sensors)  # copy
    new_env._env.sim._action_space = copy.deepcopy(old_env._env.sim._action_space)
    # new_env._env.sim._async_draw_agents_ids = copy.deepcopy(old_env._env.sim._async_draw_agents_ids)  # copy  # no puedo copiarlo
    new_env._env.sim._current_scene = copy.deepcopy(old_env._env.sim._current_scene)
    # new_env._env.sim._default_agent = copy.copy(old_env._env.sim._default_agent)  # copy  # no puedo copiarlo
    new_env._env.sim._default_agent_id = copy.deepcopy(old_env._env.sim._default_agent_id)
    new_env._env.sim._initialized = copy.deepcopy(old_env._env.sim._initialized)
    new_env._env.sim._last_state = copy.deepcopy(old_env._env.sim._last_state)
    new_env._env.sim._num_total_frames = copy.deepcopy(old_env._env.sim._num_total_frames)
    new_env._env.sim._prev_sim_obs = copy.deepcopy(old_env._env.sim._prev_sim_obs)
    new_env._env.sim._previous_step_time = copy.deepcopy(old_env._env.sim._previous_step_time)
    # new_env._env.sim._sensors = copy.copy(old_env._env.sim._sensors)  # copy  # no puedo copiarlo

    # new_env._env.sim.action_space = copy.copy(old_env._env.sim.action_space)  # copy # no puedo copiarlo
    new_env._env.sim.active_dataset = copy.deepcopy(old_env._env.sim.active_dataset)
    new_env._env.sim.agents = copy.copy(old_env._env.sim.agents)  # copy
    new_env._env.sim.config = copy.copy(old_env._env.sim.config)  # copy
    # new_env._env.sim.curr_scene_name = copy.copy(old_env._env.sim.curr_scene_name)  # copy # no puedo copiarlo
    new_env._env.sim.frustum_culling = copy.deepcopy(old_env._env.sim.frustum_culling)
    # new_env._env.sim.gfx_replay_manager = copy.copy(old_env._env.sim.gfx_replay_manager)  # copy # no puedo copiarlo
    # TODO [sergio]: Esta new_env._env.sim.gpu_device es supceptible de ser cambiada o modificada entre tareas
    # TODO [sergio]: ######################## GPU selection ################################################
    # new_env._env.sim.gpu_device = copy.copy(old_env._env.sim.gpu_device)  # copy # no puedo copiarlo
    new_env._env.sim.habitat_config = copy.deepcopy(old_env._env.sim.habitat_config)
    # new_env._env.sim.metadata_mediator = copy.copy(old_env._env.sim.metadata_mediator)  # copy # no puedo copiarlo
    new_env._env.sim.navmesh_visualization = copy.deepcopy(old_env._env.sim.navmesh_visualization)
    # new_env._env.sim.pathfinder = copy.copy(old_env._env.sim.pathfinder)  # copy # no puedo copiarlo
    # new_env._env.sim.previous_step_collided = copy.copy(old_env._env.sim.previous_step_collided)  # copy # no puedo copiarlo
    # new_env._env.sim.random = copy.copy(old_env._env.sim.random)  # copy # no puedo copiarlo
    # new_env._env.sim.renderer = copy.copy(old_env._env.sim.renderer)  # copy # no puedo copiarlo
    # new_env._env.sim.semantic_color_map = copy.copy(old_env._env.sim.semantic_color_map)  # copy # no puedo copiarlo
    # new_env._env.sim.semantic_scene = copy.copy(old_env._env.sim.semantic_scene)  # copy # no puedo copiarlo
    # new_env._env.sim.sensor_suite = copy.copy(old_env._env.sim.sensor_suite)  # copy # no puedo copiarlo
    new_env._env.sim.sim_config = copy.copy(old_env._env.sim.sim_config)  # copy
    # new_env._env.sim.up_vector = copy.copy(old_env._env.sim.up_vector)  # copy # no puedo copiarlo

    return new_env


#
# def SimpleRLEnv_deepcopy(obj):
#     config_paths = copy.copy(obj.config_path)
#     result_path = copy.copy(obj.result_path)
#     copied_env = SimpleRLEnv(config_paths=config_paths, result_path=result_path)
#     return copied_env

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


class PointNavHabitatSimple(Environment):
    def __init__(self, config_paths="configs/tasks/pointnav.yaml", result_path=os.path.join("development", "images")):
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.result_path = result_path

        super().__init__()

        config = habitat.get_config(config_paths=config_paths)
        config.defrost()
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.freeze()

        self.habitat_env = RLEnv(config=config)

        ########################################################3
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """
        # self._core_env_config = config
        # self._env = HabitatEnv(config, dataset=None)
        # self._observation_space = self._env.observation_space
        # self._action_space = self._env.action_space
        # self.number_of_episodes = self._env.number_of_episodes
        # self.reward_range = self.get_reward_range()
        # RLEnv.__init__(self, config=config)
        ####################################################

        self.episode_counter = 0
        self.episode_results_path = self.result_path
        self.episode_images = []

        self.metadata = {
            'render.modes': ['rgb', 'text']
        }
        self._action_space = akro.Discrete(3)

        self._observation_space = akro.Tuple((akro.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                                              akro.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)))

        self._visualize = False

    def reset(self):
        super().reset()
        # Al final de cada episodio guardar un video del recorido del robot
        if len(self.episode_images) > 0:
            images_to_video(self.episode_images, self.episode_results_path, "trajectory")

        self.episode_images = []
        observation = self.habitat_env.reset()

        self.episode_counter += 1

        # Definir ruta paga guardar las aejecuciones
        self.episode_results_path = os.path.join(
            self.result_path, "shortest_path_example", "%02d" % self.episode_counter
        )

        if os.path.exists(self.episode_results_path):
            shutil.rmtree(self.episode_results_path)
        os.makedirs(self.episode_results_path)

        self._state = observation

        return observation

    def step(self, *args, **kwargs):
        super().step(*args)
        observation, reward, done, info = self.habitat_env.step(self, *args, **kwargs)

        # Guardar video de las ejecuciones en un archivo
        im = observation["rgb"]
        top_down_map = self._draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        self.episode_images.append(output_im)

        reward = self._get_reward(info['distance_to_goal'])
        self._state = observation
        return observation, reward, done, info

    def render(self, mode: str = "rgb", print_on_screen=False):
        image = self.habitat_env.render(mode=mode)

        if print_on_screen and mode is 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.episode_results_path, image)
            cv2.waitKey(1)

        if mode is 'text':
            print(self._state[1])

        return image

    def close(self):
        pass

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

    # habitat.RLEnv necesita asignar valores
    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self.config

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self.metadata['render.modes']

    # TODO: los dos siguientes métodos probablemente sean utiles en metalearning
    # pylint: disable=no-self-use
    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('text', print_on_screen=self._visualize))


class HM3DRLEnv(habitat.RLEnv):

    def __init__(self, config_paths="configs/RL/objectnav_hm3d_RL.yaml",
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

    # def __deepcopy__(self, memodict={}):
    #     config_paths = copy.copy(self.config_path)
    #     result_path = copy.copy(self.result_path)
    #     copied_env = type(self)(config_paths=config_paths, result_path=result_path)
    #     # memodict[id(self)] = copied_env
    #     # copy._member1 = self._member1
    #     # copy._member2 = deepcopy(self._member2, memo)
    #
    #     # TODO [sergio]: Esta copied_env._np_random es supceptible de ser cambiada o modificada entre tareas
    #     copied_env._np_random = copy.deepcopy(self._np_random)
    #     copied_env._core_env_config = copy.deepcopy(copied_env._core_env_config)
    #     copied_env = deepcopy_habitat_RLEnv(copied_env, self)
    #
    #     copied_env.action_space = copy.deepcopy(self.action_space)
    #     # TODO [sergio]: Esta copied_env.config es supceptible de ser cambiada o modificada entre tareas
    #     # copied_env.config = copy.copy(self.config)  # copy # no puedo copiarlo
    #     copied_env.config_path = copy.deepcopy(self.config_path)
    #     # copied_env.current_episode = copy.copy(self.current_episode)  # copy # no puedo copiarlo
    #     copied_env.episode_counter = copy.deepcopy(self.episode_counter)
    #     copied_env.episode_images = copy.deepcopy(self.episode_images)
    #     # TODO [sergio]: Esta copied_env.episode_results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.episode_results_path = copy.deepcopy(self.episode_results_path)  # copy # no puedo copiarlo
    #     copied_env.episodes = copy.deepcopy(self.episodes)
    #     # copied_env.habitat_env = copy.copy(self.habitat_env)  # copy # no puedo copiarlo
    #     copied_env.metadata = copy.deepcopy(self.metadata)
    #     copied_env.number_of_episodes = copy.deepcopy(self.number_of_episodes)
    #     copied_env.observation_space = copy.deepcopy(self.observation_space)
    #     # TODO [sergio]: Esta copied_env.results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.result_path = copy.deepcopy(self.result_path)
    #     copied_env.reward_range = copy.deepcopy(self.reward_range)
    #     copied_env.spec = copy.deepcopy(self.spec)
    #
    #     return copied_env


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

    # def __deepcopy__(self, memodict={}):
    #     config_paths = copy.copy(self.config_path)
    #     result_path = copy.copy(self.result_path)
    #     copied_env = type(self)(config_paths=config_paths, result_path=result_path)
    #     # memodict[id(self)] = copied_env
    #     # copy._member1 = self._member1
    #     # copy._member2 = deepcopy(self._member2, memo)
    #
    #     # TODO [sergio]: Esta copied_env._np_random es supceptible de ser cambiada o modificada entre tareas
    #     copied_env._np_random = copy.deepcopy(self._np_random)
    #     copied_env._core_env_config = copy.deepcopy(copied_env._core_env_config)
    #     copied_env = deepcopy_habitat_RLEnv(copied_env, self)
    #
    #     copied_env.action_space = copy.deepcopy(self.action_space)
    #     # TODO [sergio]: Esta copied_env.config es supceptible de ser cambiada o modificada entre tareas
    #     # copied_env.config = copy.copy(self.config)  # copy # no puedo copiarlo
    #     copied_env.config_path = copy.deepcopy(self.config_path)
    #     # copied_env.current_episode = copy.copy(self.current_episode)  # copy # no puedo copiarlo
    #     copied_env.episode_counter = copy.deepcopy(self.episode_counter)
    #     copied_env.episode_images = copy.deepcopy(self.episode_images)
    #     # TODO [sergio]: Esta copied_env.episode_results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.episode_results_path = copy.deepcopy(self.episode_results_path)  # copy # no puedo copiarlo
    #     copied_env.episodes = copy.deepcopy(self.episodes)
    #     # copied_env.habitat_env = copy.copy(self.habitat_env)  # copy # no puedo copiarlo
    #     copied_env.metadata = copy.deepcopy(self.metadata)
    #     copied_env.number_of_episodes = copy.deepcopy(self.number_of_episodes)
    #     copied_env.observation_space = copy.deepcopy(self.observation_space)
    #     # TODO [sergio]: Esta copied_env.results_path es supceptible de ser cambiada o modificada entre tareas
    #     copied_env.result_path = copy.deepcopy(self.result_path)
    #     copied_env.reward_range = copy.deepcopy(self.reward_range)
    #     copied_env.spec = copy.deepcopy(self.spec)
    #
    #     return copied_env
