from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os


def smooth(scalars: List[float], weight: float = 0.5) -> List[float]:  # Weight between 0 and 1
    """
    Tensorboard smoothing
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


graph_type = 'steps_per_episode'
files = os.listdir(graph_type)
data = pd.DataFrame()
exp_names = {
    'S32': 'S3 w/ rew shaping and exploration',
    'S52': 'S5 w/ rew shaping and exploration',
    'S55': 'S5 w/ rew shaping and no exploration',
    'S58': 'S5 w/ sparse rew and exploration',
    'S511': 'S5 w/ sparse rew and no exploration',
}

for f in files:
    exp_name = f.split('_')[0] + f.split('_')[1]
    df = pd.read_csv(os.path.join(graph_type, f))
    plt.plot(df['Step'], smooth(df['Value'], weight=0.99), label=exp_names[exp_name])

plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 6))
plt.title(graph_type)
plt.legend()
plt.grid()
plt.xlim([0, 10000000])
plt.show()
