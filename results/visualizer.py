import cloudpickle
import matplotlib.pyplot as plt

filename = open('test_envs.pkl', 'rb')
envs = cloudpickle.load(filename)

for env in envs:
    plt.axis("off")
    plt.imshow(env.render_top_view())
    plt.show()
