import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(400)]

epsilon_init = 1.0
epsilon = 1.
y = []
for i in range(400):
    y.append(epsilon)
    epsilon = epsilon*0.95
    if i % 100==0:
        # epsilon_init = epsilon_init - 0.2
        epsilon = epsilon_init


plt.plot(x, y)
plt.ylabel('épsilon')
plt.xlabel('iteraciones')
plt.show()

