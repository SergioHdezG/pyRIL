import pygame
import pandas as pd
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt


class Callbacks:
    def __init__(self):
        self.memory = []
        self.counter = 0
        self.max_iter = 500

    def remember_callback(self, obs, next_obs, action, reward, done, info=None):
        self.counter += 1
        self.memory.append([obs, action, reward, next_obs, done])
        # print(self.counter)
        # if self.counter >= self.max_iter:
        #     try:
        #         event = pygame.event.Event(pygame.QUIT)
        #         pygame.event.post(event)
        #     except:
        #         pass
        # print("Pygame is trying to exit the game")

    def reset_memory(self):
        self.memory = []

    def memory_to_csv(self, dir, name):
        if len(self.memory) > 0:
            obs = [x[0] for x in self.memory]
            act = [x[1] for x in self.memory]
            reward = np.array([x[2] for x in self.memory])
            next_obs = np.array([x[3] for x in self.memory])
            done = np.array([x[4] for x in self.memory])

            data = pd.DataFrame({'obs': obs, 'action': act, 'reward': reward, 'done': done})
            print('data ready')
            # data = pd.DataFrame({'obs': obs, 'action': act, 'reward': reward, 'next_obs': next_obs, 'done': done })
            # data.to_pickle(dir+name+'.pkl')
            data.to_csv(dir + name + '.csv')
            print('data saved')

    def save_memories(self, path):
        if len(self.memory) > 0:
            obs = [x[0] for x in self.memory]
            act = [x[1] for x in self.memory]
            reward = np.array([x[2] for x in self.memory])
            next_obs = np.array([x[3] for x in self.memory])
            done = np.array([x[4] for x in self.memory])

            data = pd.DataFrame({'obs': obs, 'action': act, 'reward': reward, 'done': done})
            print('data ready')
            ext = os.path.splitext(path)[-1].lower()

            if ext != ".pkl" and ext != ".csv":
                raise Exception("File extension " + str(ext) + " do not match .pkl or .csv file formats.")

            folder = os.path.dirname(path)
            if not os.path.exists(folder) and folder != '':
                os.makedirs(folder)
            data.to_pickle(path)
            print('data saved')

    def get_memory(self, get_action, n_stack=0):
        if len(self.memory) > 0:
            obs = np.array([x[0] for x in self.memory])
            action = np.array([x[1] for x in self.memory])
            done = np.array([x[4] for x in self.memory])

            # if get_action:
            #     data = []
            #     for i in range(obs.shape[0]):
            #         data.append(np.array([obs[i], np.reshape(action[i], (1,))]))
            #     return np.array(data)
            data = format_obs(obs, action, get_action, done, n_stack)
            return data

    def set_max_iter(self, iterations):
        self.max_iter = iterations


def load_expert_memories(path, load_action, n_stack=0, not_formated=False, img_data=False):
    """
    :param not_formated: bool. If true, loaded data won't be formated. If false, loaded data will be formated depending
        on n_stack value.
    """
    # data = pd.read_csv(dir+name+".csv")
    data = pd.read_pickle(path)
    # print(data.head())
    obs = data['obs'].values
    action = data['action'].values
    reward = data['reward'].values
    done = data['done'].values

    obs = np.array([np.array(x) for x in obs])
    action = np.array([np.array(x) for x in action])
    reward = np.array([np.array(x) for x in reward])
    done = np.array([np.array(x) for x in done])

    if not_formated:
        data = [np.array([o, a, r, d]) for o, a, r, d in zip(obs, action, reward, done)]
        return data
    else:
        data = format_obs(obs, action, load_action, done, n_stack, img_data)
        return data


def format_obs(obs, action, get_action, done, n_stack=0, img_data=False):

    if img_data:
        if n_stack > 1:
            data = []
            obs_stack = deque(maxlen=n_stack)

            first_obs = True
            for i in range(0, obs.shape[0]):
                if first_obs:
                    for _ in range(n_stack):
                        obs_stack.append(np.array(obs[i]))
                else:
                    obs_stack.append(np.array(obs[i]))

                o_s = np.squeeze(obs_stack.copy(), axis=-1)
                o_s = o_s.transpose((1, 2, 0))
                if get_action:
                    # data.append([np.reshape(obs_stack.copy(), (-1, obs_stack.maxlen)), action[i]])
                    # data.append([np.transpose(obs_stack.copy()), action[i]])
                    # ssss = np.array(obs_stack.copy())
                    data.append([np.array(o_s), action[i]])
                else:
                    data.append(o_s)
                first_obs = done[i]  # Trata de tener en cuenta las primeras observaciones de un episodio,
                # pero depende de como se hayan recopilado los datos

        else:
            if get_action:
                data = []
                for i in range(obs.shape[0]):
                    # data.append(np.array([obs[i], [action[i]], reward[i], done[i]]))
                    data.append(np.array([obs[i], action[i]], dtype=object))
                    # data.append([obs[i], action[i]])

            else:
                data = []
                for i in range(obs.shape[0]):
                    # data.append(np.array([obs[i]], dtype=object))
                    data.append(np.array(obs[i], dtype=np.float32))
    else:
        if n_stack > 1:
            data = []
            obs_stack = deque(maxlen=n_stack)

            first_obs = True
            for i in range(0, obs.shape[0]):
                if first_obs:
                    for _ in range(n_stack):
                        obs_stack.append(np.array(obs[i]))
                else:
                    obs_stack.append(np.array(obs[i]))

                if get_action:
                    # data.append([np.reshape(obs_stack.copy(), (-1, obs_stack.maxlen)), action[i]])
                    # data.append([np.transpose(obs_stack.copy()), action[i]])
                    # ssss = np.array(obs_stack.copy())
                    data.append([np.array(obs_stack.copy()), action[i]])
                else:
                    data.append(obs_stack.copy())
                first_obs = done[i]  # Trata de tener en cuenta las primeras observaciones de un episodio,
                # pero depende de como se hayan recopilado los datos

        else:
            if get_action:
                data = []
                for i in range(obs.shape[0]):
                    # data.append(np.array([obs[i], [action[i]], reward[i], done[i]]))
                    data.append(np.array([obs[i], action[i]], dtype=object))
                    # data.append([obs[i], action[i]])

            else:
                data = []
                for i in range(obs.shape[0]):
                    # data.append(np.array([obs[i]], dtype=object))
                    data.append(np.array([obs[i]], dtype=np.float32))

    return data

def dummy_preprocess(obs):
    return obs
