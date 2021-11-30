import itertools
import math
import random

import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from gym import spaces

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RoomNavi(gym.Env):
    def __init__(self, num_cells=3, corridor_len=3, num_steps=50):
        super(RoomNavi, self).__init__()

        self.seed()
        self.num_cells = num_cells  # should be even!
        self.corridor_len = corridor_len
        self.num_states = 3 * (self.num_cells ** 2) + 2 * self.corridor_len

        self.width = 3 * self.num_cells + 2 * self.corridor_len
        self.height = self.num_cells

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.observation_space = spaces.Box(low=0, high=3 * self.num_cells + 2 * self.corridor_len - 1, shape=(2,))
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 6  # location of goal 1, goal 2, goal 3
        self.num_tasks = 4 * 4 * 4  # loc of goal 1 (4 options) x loc of goal 2 (4 options) x loc of goal 3 (4 options)

        # starting state (middle)
        self.starting_state = (0.0, 0.0)
        
        # goals can be anywhere except on possible starting states and immediately around it
        self.offset = (self.num_cells - 1) // 2
        self.possible_goals_middle = [[-self.offset, self.offset],
                                      [-self.offset, -self.offset],
                                      [self.offset, self.offset],
                                      [self.offset, -self.offset]]
        middle_left = -self.corridor_len - 2 * self.offset - 1
        self.possible_goals_left = [[middle_left - self.offset, self.offset],
                                    [middle_left - self.offset, -self.offset],
                                    [middle_left + self.offset, self.offset],
                                    [middle_left + self.offset, -self.offset]]
        middle_right = self.corridor_len + 2 * self.offset + 1
        self.possible_goals_right = [[middle_right - self.offset, self.offset],
                                     [middle_right - self.offset, -self.offset],
                                     [middle_right + self.offset, self.offset],
                                     [middle_right + self.offset, -self.offset]]

        self.possible_states = set(itertools.product(range(-self.num_cells - self.corridor_len - self.offset,
                                                           self.num_cells + self.corridor_len + self.offset + 1),
                                                     range(-self.offset, self.offset+1)))
        remove_left = set(itertools.product(range(-self.corridor_len - self.offset, -self.offset),
                                            [*list(range(-self.num_cells+self.offset, 0)),
                                             *list(range(1, self.num_cells-self.offset))]
                                            ))
        remove_right = set(itertools.product(range(self.offset + 1, self.corridor_len + self.offset + 1),
                                             [*list(range(-self.num_cells+self.offset, 0)),
                                              *list(range(1, self.num_cells-self.offset))]
                                             ))
        self.possible_states = self.possible_states.difference(remove_left)
        self.possible_states = self.possible_states.difference(remove_right)
        self.possible_states = list(self.possible_states)
        self.index_matrix = torch.zeros((self.height, self.width)).long()
        for i, (x, y) in enumerate(self.possible_states):
            self.index_matrix[y + self.height//2,
                              x + self.width//2] = i

        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._goals = self.reset_task()

    def reset_task(self, task=None):
        if task is None:
            self.goal_1 = random.choice(self.possible_goals_middle)
            if self.goal_1[0] < 0:  # left
                self.goal_2 = random.choice(self.possible_goals_left)
            else:
                self.goal_2 = random.choice(self.possible_goals_right)
            self.goal_3 = random.choice(self.possible_goals_middle)
            while self.goal_3 == self.goal_1:
                self.goal_3 = random.choice(self.possible_goals_middle)
        else:
            self.goal_1 = task[:2]
            self.goal_2 = task[2:4]
            self.goal_3 = task[4:]
        self._goals = np.array([*self.goal_1, *self.goal_2, *self.goal_3])
        self.reached_goal_1 = False
        self.reached_goal_2 = False
        self.reached_goal_3 = False
        return self._goals

    def get_task(self):
        return self._goals.copy()

    def reset(self):
        self.step_count = 0
        self.reached_goal_1 = False
        self.reached_goal_2 = False
        self.reached_goal_3 = False
        self._env_state = np.array(self.starting_state)
        return self._env_state.copy()

    def state_transition(self, action):
        """
        Moving the agent between states
        """

        # -- CASE: INSIDE CORRIDOR --

        inside_left_corridor = - self.corridor_len - self.offset <= self._env_state[0] <= - self.offset - 1
        inside_right_corridor = self.offset + 1 <= self._env_state[0] <= self.corridor_len + self.offset

        if inside_left_corridor or inside_right_corridor:
            if action in [1, 3]:  # up or down
                # cannot walk into walls
                return self._env_state
            else:
                # left and right is always possible
                if action == 2:  # right
                    self._env_state[0] = self._env_state[0] + 1
                elif action == 4:  # left
                    self._env_state[0] = self._env_state[0] - 1
                return self._env_state

        # -- CASE: INSIDE ROOM --

        # execute action to see where we'd end up
        new_env_state = self._env_state.copy()
        if action == 1:  # up
            new_env_state[1] = new_env_state[1] + 1
        elif action == 2:  # right
            new_env_state[0] = new_env_state[0] + 1
        elif action == 3:  # down
            new_env_state[1] = new_env_state[1] - 1
        elif action == 4:  # left
            new_env_state[0] = new_env_state[0] - 1

        walked_into_left_corridor = - self.corridor_len - self.offset <= new_env_state[0] <= - self.offset - 1
        walked_into_right_corridor = self.offset + 1 <= new_env_state[0] <= self.corridor_len + self.offset

        # check if this is a valid spot
        if walked_into_left_corridor or walked_into_right_corridor:
            if new_env_state[1] == 0:
                self._env_state = new_env_state
        else:
            inside_x = -self.corridor_len - self.num_cells - self.offset <= new_env_state[0] <= self.corridor_len + self.num_cells + self.offset
            indside_y = -self.num_cells+self.offset < new_env_state[1] < self.num_cells-self.offset
            if inside_x and indside_y:
                self._env_state = new_env_state

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        # assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute reward
        if self._env_state[0] == self.goal_1[0] and self._env_state[1] == self.goal_1[1]:
            self.reached_goal_1 = True
            reward = 1.0
        elif self.reached_goal_1 and self._env_state[0] == self.goal_2[0] and self._env_state[1] == self.goal_2[1]:
            self.reached_goal_2 = True
            reward = 10.0
        elif self.reached_goal_2 and self._env_state[0] == self.goal_3[0] and self._env_state[1] == self.goal_3[1]:
            self.reached_goal_3 = True
            reward = 100.0
        else:
            reward = -0.1

        task = self.get_task()
        info = {'task': task}
        return state, reward, done, info

    def obs_to_state_idx(self, cell):
        if isinstance(cell, list) or isinstance(cell, tuple):
            cell = np.array(cell)
        if isinstance(cell, np.ndarray):
            cell = torch.from_numpy(cell)
        cell = cell.long()
        cell_shape = cell.shape
        if len(cell_shape) > 2:
            cell = cell.reshape(-1, cell.shape[-1])
        indices = self.index_matrix[cell[:, 1], cell[:, 0]]
        indices = indices.reshape(cell_shape[:-1])
        return indices

    def state_idx_to_obs(self, idx):
        return self.possible_states[idx]

    def task_to_id(self, goals):

        combinations_left = list(itertools.product(self.possible_goals_middle[:2], self.possible_goals_left))
        combinations_right = list(itertools.product(self.possible_goals_middle[2:], self.possible_goals_right))
        combinations = [*combinations_left, *combinations_right]

        combinations = np.array(combinations).reshape(-1, 4)
        mask = np.sum(np.abs(combinations - goals), axis=-1) == 0

        classes = torch.arange(0, len(mask))[mask].item()

        return classes

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            **kwargs
                            ):
        """
        Visualises the behaviour of the policy, together with the latent state and belief.
        The environment passed to this method should be a SubProcVec or DummyVecEnv, not the raw env!
        """

        num_episodes = args.max_rollouts_per_task

        # --- initialise things we want to keep track of ---

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episode_goals = []
        if args.pass_belief_to_policy and (encoder is None):
            episode_beliefs = [[] for _ in range(num_episodes)]
        else:
            episode_beliefs = None

        if encoder is not None:
            # keep track of latent spaces
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        # --- roll out policy ---

        env.reset_task()
        [state, belief, task] = utl.reset_env(env, args)
        start_obs = state.clone()

        for episode_idx in range(args.max_rollouts_per_task):

            curr_goal = env.get_task()
            curr_rollout_rew = []
            curr_rollout_goal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_all_obs[episode_idx].append(start_obs.clone())
            if args.pass_belief_to_policy and (encoder is None):
                episode_beliefs[episode_idx].append(belief)

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())

                # act
                _, action = utl.select_action(args=args,
                                                 policy=policy,
                                                 state=state.view(-1),
                                                 belief=belief,
                                                 task=task,
                                                 deterministic=True,
                                                 latent_sample=curr_latent_sample.view(-1) if (curr_latent_sample is not None) else None,
                                                 latent_mean=curr_latent_mean.view(-1) if (curr_latent_mean is not None) else None,
                                                 latent_logvar=curr_latent_logvar.view(-1) if (curr_latent_logvar is not None) else None,
                                                 )

                # observe reward and next obs
                [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(env, action, args)

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.float().to(device),
                        state,
                        rew_raw.reshape((1, 1)).float().to(device),
                        hidden_state,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(state.clone())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())
                curr_rollout_goal.append(env.get_task().copy())

                if args.pass_belief_to_policy and (encoder is None):
                    episode_beliefs[episode_idx].append(belief)

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1)).to(device)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episode_goals.append(curr_goal)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot behaviour
        num_cells = env.get_env_attr('num_cells')
        horizon = env.get_env_attr('_max_episode_steps')

        subplot_indices = [1, 4 * num_cells, horizon]
        plt.figure(figsize=(5, 1.5 * len(subplot_indices)))

        num_episodes = len(episode_all_obs)
        num_steps = len(episode_all_obs[0])

        # loop through the experiences
        for episode_idx in range(num_episodes):
            for step_idx in range(num_steps):

                curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
                curr_goal = episode_goals[episode_idx]

                if step_idx in subplot_indices:
                    # choose correct subplot
                    plt.subplot(len(subplot_indices), 1, subplot_indices.index(step_idx)+1)
                    # plot the behaviour
                    plot_behaviour(env, curr_obs, curr_goal)
                    if episode_idx == 0:
                        plt.title('t = {}'.format(step_idx))

        # save figure that shows policy behaviour
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns


def plot_behaviour(env, observations, goal):

    corridor_len = env.get_env_attr('corridor_len')
    num_cells = env.get_env_attr('num_cells')
    offset = env.get_env_attr('offset')
    
    width = 3 * num_cells + 2 * corridor_len
    height = num_cells

    # draw grid
    for x in range(width):
        for y in range(height):

            orig_x = x - width // 2
            orig_y = y - height // 2

            # check if this is a valid spot
            facecolor = 'k'
            # - corridors
            inside_left_corridor = - corridor_len - offset <= orig_x <= - offset - 1
            inside_right_corridor = offset + 1 <= orig_x <= corridor_len + offset
            if inside_left_corridor or inside_right_corridor:
                if orig_y == 0:
                    facecolor = 'none'
            else:
                # - cells left/right
                inside_x = -corridor_len - num_cells - offset <= orig_x <= corridor_len + num_cells + offset
                inside_y = -height+offset < orig_y < height-offset
                if inside_x and inside_y:
                    facecolor = 'none'

            rec = Rectangle((x, y), 1, 1, facecolor=facecolor, alpha=1.0, edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # shift obs and goal so that their origin is on the bottom left
    observations[:, 0] += width // 2
    observations[:, 1] += height // 2
    goal[[0, 2, 4]] += width // 2
    goal[[1, 3, 5]] += height // 2

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')
    plt.plot(goal[2], goal[3], 'k*')
    plt.plot(goal[4], goal[5], 'ko')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 3 * num_cells + 2 * corridor_len])
    plt.ylim([0, num_cells])
