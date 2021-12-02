import numpy as np

def discount_and_norm_rewards(rewards, mask, gamma, norm=True, n_step_return=None):
    """
    Calculate the return as cumulative discounted rewards of an episode.
    :param episode_rewards: ([float]) List of rewards of an episode.
    """
    discounted_episode_rewards = np.zeros_like(rewards)

    # Calculate cumulative returns for n-steps of each trajectory
    if n_step_return is not None:
        cumulative_return = 0
        for i in reversed(range(rewards.size-n_step_return, rewards.size)):
            cumulative_return = rewards[i] + cumulative_return * gamma * mask[i]
            discounted_episode_rewards[i] = cumulative_return

        for i in reversed(range(rewards.size-n_step_return)):
            cumulative_return = 0
            for j in reversed(range(i, i + n_step_return)):
                cumulative_return = rewards[j] + cumulative_return * gamma * mask[j]
            discounted_episode_rewards[i] = cumulative_return
    else:
        cumulative_return = 0
        # Calculate cumulative returns for entire trajectories
        for i in reversed(range(rewards.size)):
            cumulative_return = rewards[i] + cumulative_return * gamma * mask[i]
            discounted_episode_rewards[i] = cumulative_return

    if norm:
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards) + 1e-10  # para evitar valores cero
    return discounted_episode_rewards


rew = np.array([2., 6., 8., 1., 0., -10., 5., -2., 4.])
mask = np.array([True, True, True, True, False, True, True, True, True])
gamma = 0.9
n_step_return = 3

result = discount_and_norm_rewards(rew, mask, gamma, norm=False, n_step_return=None)

print(result)
