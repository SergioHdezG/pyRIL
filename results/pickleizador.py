import cloudpickle
import matplotlib.pyplot as plt
from environments.maze import PyMaze

envs = []
for _ in range(10):
    env = PyMaze()
    plt.imshow(env.render_top_view())
    plt.show()
    envs.append(env)

filename = 'test_envs.pkl'
outfile = open(filename, 'wb')
cloudpickle.dump(envs, outfile)
outfile.close()
