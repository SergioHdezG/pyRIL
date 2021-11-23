import numpy as np
import matplotlib.pyplot as plt
img = np.load("/home/serch/TFM/IRL3/environments/carla/dataset/rgb.npy")

plt.figure(1)
plt.imshow(img[0])
plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure(1, figsize=(20, 4))
# x = [i for i in range(400)]
#
# epsilon_init = 1.0
# epsilon = 1.
# y = []
# for i in range(400):
#     y.append(epsilon)
#     epsilon = epsilon*0.99
#
# ax = fig.add_subplot(1, 3, 1)
# ax.plot(x, y)
# ax.set(xlabel='Iterations', ylabel='Epsilon', title='Descending')
#
# epsilon_init = 1.0
# epsilon = 1.
# y = []
# for i in range(400):
#     y.append(epsilon)
#     epsilon = epsilon*0.95
#     if i % 100==0:
#         epsilon = epsilon_init
#
# ax = fig.add_subplot(1, 3, 2)
# ax.plot(x, y)
# ax.set(xlabel='Iterations', title='Cyclical')
#
# epsilon_init = 1.0
# epsilon = 1.
# y = []
# for i in range(400):
#     y.append(epsilon)
#     epsilon = epsilon*0.95
#     if i % 100==0:
#         epsilon_init = epsilon_init - 0.2
#         epsilon = epsilon_init
#
# ax = fig.add_subplot(1, 3, 3)
# ax.plot(x, y)
# ax.set(xlabel='Iterations', title='Cyclic-Descending')
#
# plt.show()
