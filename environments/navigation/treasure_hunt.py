import random
import matplotlib.colors as mcolors
import numpy as np
import torch
import matplotlib.pyplot as plt

from gym import Env
from gym import spaces
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TreasureHunt(Env):
    def __init__(self,
                 max_episode_steps=100,
                 mountain_height=1,
                 treasure_reward=1,
                 timestep_penalty=-1,
                 ):

        # environment layout
        self.mountain_top = np.array([0, 0])
        self.start_position = np.array([0, -1])
        self.mountain_height = mountain_height
        self.treasure_reward = treasure_reward
        self.timestep_penalty = timestep_penalty
        # NOTE: much of the code assumes these radii, if you change them you need to make a few other changes!
        self.goal_radius = 1.0
        self.mountain_radius = 0.5
        # You can get around the full circle in 62 steps
        self._max_episode_steps = max_episode_steps

        # observation/action/task space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.task_dim = 2

        # initialise variables
        self.step_count = None
        self.goal = None
        self.state = None
        self.reset_task()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.state = self.start_position
        return self._get_obs()

    def sample_task(self):
        # sample a goal from a circle with radius 1 around the mountain top [0, 0]
        angle = random.uniform(0, 2 * np.pi)
        goal = self.goal_radius * np.array((np.cos(angle), np.sin(angle)))
        return goal

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        # execute action - make sure the agent does not walk outside [-1.5, 1.5] in any direction
        self.state = np.clip(self.state + 0.1 * action, -1.5, 1.5)
        done = False

        # if the agent is on the goal, it gets a high reward
        mountain_top_distance = np.linalg.norm(self.mountain_top - self.state, 2)
        treasure_distance = np.linalg.norm(self.goal - self.state, 2)

        # CASE: agent is on mountain - penalise (the higher the more penalty)
        if mountain_top_distance <= self.mountain_radius:
            reward = self.mountain_height * (- self.mountain_radius + mountain_top_distance) + self.timestep_penalty
        # CASE: agent is near the goal - give it the treasure reward
        elif treasure_distance <= 0.1:
            reward = self.treasure_reward
        # CASE: agent is somewhere else - make sure it doesn't walk too far away
        else:
            # make agent not walk too far outside circle
            dist_to_center = max([1, np.sqrt(self.state[0]**2 + self.state[1]**2)])
            reward = self.timestep_penalty * dist_to_center

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def set_task(self, task):
        self.goal = task

    def get_task(self):
        return self.goal

    def _get_obs(self):
        agent_is_on_mountain = np.linalg.norm(self.state, 2) < 0.1
        if agent_is_on_mountain:
            obs = np.concatenate((self.state, self.goal))
        else:
            obs = np.concatenate((self.state, np.zeros(2)))
        return obs

    def render(self, mode='human'):
        pass

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            **kwargs):

        policy = policy.actor_critic
        encoder = encoder

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0]

        # --- initialise things we want to keep track of ---

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]
        episode_tasks = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episodegoals = []
        if getattr(unwrapped_env, 'belief_oracle', False):
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
        obs, belief, task = utl.reset_env(env, args)
        obs = obs.float().reshape((1, -1)).to(device)
        start_obs = obs.clone()
        for episode_idx in range(args.max_rollouts_per_task):

            currgoal = env.get_task()

            curr_rollout_rew = []
            curr_rolloutgoal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hiddenstate = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0]
                    curr_latent_mean = curr_latent_mean[0]
                    curr_latent_logvar = curr_latent_logvar[0]

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            if getattr(unwrapped_env, 'belief_oracle', False):
                episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._beliefstate.copy())

            episode_all_obs[episode_idx].append(start_obs.clone())
            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                else:
                    episode_prev_obs[episode_idx].append(obs.clone())

                # act
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=obs,
                                              deterministic=True,
                                              latent_sample=curr_latent_sample, latent_mean=curr_latent_mean,
                                              latent_logvar=curr_latent_logvar,
                                              task=task)

                # observe reward and next obs
                [obs, _, task], (rew, rew_normalised), done, infos = utl.env_step(env, action, args)
                obs = obs.reshape((1, -1)).to(device)

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hiddenstate = encoder(
                        action.view((1, -1)).float(),
                        obs,
                        rew.reshape((1, 1)).float(),
                        hiddenstate,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(obs.clone())
                episode_next_obs[episode_idx].append(obs.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())
                episode_tasks[episode_idx].append(task.clone())

                curr_rollout_rew.append(rew.clone())
                curr_rolloutgoal.append(env.get_task().copy())

                if getattr(unwrapped_env, 'belief_oracle', False):
                    episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._beliefstate.copy())

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['startstate']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1))
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episodegoals.append(currgoal)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot behaviour & visualise belief in env
        # plot the mountain
        angle = np.linspace(0, 2 * np.pi, 100)
        n = 7
        m = 15
        cmap = plt.get_cmap('viridis', n+m)
        for j, r in enumerate(np.linspace(0.1, 0.5, n)):
            plt.plot(r * np.cos(angle), r * np.sin(angle), '-', linewidth=2,
                     color=cmap(n+m-j), # 5,6,7,...,14
                     )
        # plot the possible goals
        angle = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(angle), np.sin(angle), '--', color=mcolors.CSS4_COLORS['limegreen'], linewidth=10.2, alpha=0.2)
        # plot the goal
        plt.plot(env.get_task()[0], env.get_task()[1], 'x', color=mcolors.CSS4_COLORS['black'], markersize=25, markeredgewidth=4)
        # plot the trajectory (on top of everything)
        plt.plot(episode_prev_obs[0][:, 0].cpu(), episode_prev_obs[0][:, 1].cpu(), '-', alpha=0.3, linewidth=5, color=mcolors.CSS4_COLORS['royalblue'])
        plt.plot(episode_prev_obs[0][:, 0].cpu(), episode_prev_obs[0][:, 1].cpu(), '.', markersize=15, color=mcolors.CSS4_COLORS['blue'])
        # save figure that shows policy behaviour
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.gca().set_aspect('equal', 'box')
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns

    def visualise_reward_bonus(self,
                               env,
                               args,
                               iter_idx,
                               policy,
                               encoder=None,
                               vae=None,
                               image_folder=None,
                               intrinsic_reward=None,
                               **kwargs):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0]

        # --- initialise things we want to keep track of ---

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]
        episode_tasks = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episodegoals = []
        episode_beliefs = [[] for _ in range(num_episodes)]

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
        obs, belief, task = utl.reset_env(env, args)
        obs = obs.float().reshape((1, -1)).to(device)
        start_obs = obs.clone()

        for episode_idx in range(args.max_rollouts_per_task):

            currgoal = env.get_task()

            curr_rollout_rew = []
            curr_rolloutgoal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hiddenstate = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0]
                    curr_latent_mean = curr_latent_mean[0]
                    curr_latent_logvar = curr_latent_logvar[0]

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            if getattr(unwrapped_env, 'belief_oracle', False):
                episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._beliefstate.copy())

            episode_all_obs[episode_idx].append(start_obs.clone())
            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                else:
                    episode_prev_obs[episode_idx].append(obs.clone())

                # act
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=obs,
                                              deterministic=True,
                                              latent_sample=curr_latent_sample, latent_mean=curr_latent_mean,
                                              latent_logvar=curr_latent_logvar,
                                              task=task)

                # observe reward and next obs
                [obs, _, task], (rew, rew_normalised), done, infos = utl.env_step(env, action, args)
                obs = obs.reshape((1, -1)).to(device)
                action = action.reshape((1, -1)).to(device)

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hiddenstate = encoder(
                        action.float(),
                        obs,
                        rew.reshape((1, 1)).float(),
                        hiddenstate,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(obs.clone())
                episode_next_obs[episode_idx].append(obs.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())
                episode_tasks[episode_idx].append(task.clone())

                curr_rollout_rew.append(rew.clone())
                curr_rolloutgoal.append(env.get_task().copy())

                if getattr(unwrapped_env, 'belief_oracle', False):
                    episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._beliefstate.copy())

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['startstate']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1))
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episodegoals.append(currgoal)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        episode_tasks = [torch.stack(e) for e in episode_tasks]

        # -------------------------------------

        # compute VAE loss

        num_elbos = len(episode_latent_means[0]) - 1
        num_decodes = len(episode_prev_obs[0])

        # expand the state/rew/action inputs to the decoder (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_prev_obs = episode_prev_obs[0].unsqueeze(0).expand((num_elbos, *episode_prev_obs[0].shape))
        dec_next_obs = episode_next_obs[0].unsqueeze(0).expand((num_elbos, *episode_next_obs[0].shape))
        dec_actions = episode_actions[0].unsqueeze(0).expand((num_elbos, *episode_actions[0].shape))

        # expand the latent (to match the number of state/rew/action inputs to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        latent_means = episode_latent_means[0][:-1]
        latent_logvars = episode_latent_logvars[0][:-1]
        latent_samples = utl.sample_gaussian(latent_means, latent_logvars)
        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

        all_rewards = []
        all_reward_names = []

        # # ---> state reconstruction
        if args.decode_state:
            loss_state, state_pred = vae.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs, dec_actions, return_predictions=True)
        else:
            state_pred = None

        if args.add_exploration_bonus:
            _, bonus_state, _, bonus_hyperstate, bonus_vae = intrinsic_reward.reward(state=episode_next_obs[0],
                                                                                     belief=torch.cat((latent_means, latent_logvars), dim=-1),
                                                                                     vae=vae,
                                                                                     latent_mean=[latent_means],
                                                                                     latent_logvar=[latent_logvars],
                                                                                     batch_prev_obs=episode_prev_obs[0].unsqueeze(0),
                                                                                     batch_next_obs=episode_next_obs[0].unsqueeze(0),
                                                                                     batch_actions=episode_actions[0].unsqueeze(0),
                                                                                     batch_rewards=episode_rewards[0].unsqueeze(0),
                                                                                     batch_tasks=episode_tasks[0].unsqueeze(0),
                                                                                     return_individual=True)

            if not isinstance(bonus_state, int):
                all_rewards.append(bonus_state.view(-1))
                all_reward_names.append('bonus_state')
            if not isinstance(bonus_hyperstate, int):
                all_rewards.append(bonus_hyperstate.view(-1))
                all_reward_names.append('bonus_hyperstate')
            if not isinstance(bonus_vae, int):
                all_rewards.append(bonus_vae.view(-1))
                all_reward_names.append('bonus_vae')

        plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        n = 20
        if state_pred is not None:
            sp = state_pred[range(self._max_episode_steps), range(self._max_episode_steps)].detach().cpu().numpy()
            plt.plot(range(n), sp[:, 0][:n], label='predicted pos')
            plt.plot(range(n), sp[:, 2][:n], label='predicted goal')
        episode_next_obs[0] = episode_next_obs[0].detach().cpu().numpy()
        plt.plot(range(n), episode_next_obs[0][:, 0][:n], label='actual pos')
        plt.plot(range(n), episode_next_obs[0][:, 2][:n], label='actual goal')
        plt.legend()
        plt.title('x')
        plt.subplot(2, 2, 2)
        plt.plot(range(n), episode_next_obs[0][:, 1][:n], label='actual pos')
        if state_pred is not None:
            plt.plot(range(n), sp[:, 1][:n], label='predicted pos')
            plt.plot(range(n), sp[:, 3][:n], label='predicted goal')
        plt.plot(range(n), episode_next_obs[0][:, 3][:n], label='actual goal')
        if 'bonus_hyperstate' in all_reward_names:
            plt.subplot(2, 2, 3)
            plt.plot(bonus_hyperstate.cpu())
            plt.title('hyperstate bonus')
        elif 'bonus_state' in all_reward_names:
            plt.subplot(2, 2, 3)
            plt.plot(bonus_state.cpu())
            plt.title('state bonus')
        if 'bonus_vae' in all_reward_names:
            plt.subplot(2, 2, 4)
            plt.plot(bonus_vae.cpu())
            plt.title('vae bonus')
        plt.legend()
        plt.title('y')
        plt.savefig(f'{image_folder}/{iter_idx}_bonus_summary')
        plt.close()

        for b, curr_rew in enumerate(all_rewards):

            try:
                curr_rew = curr_rew.detach().cpu().numpy()
            except:
                curr_rew = np.array(curr_rew)

            # plot reward bonuses over time
            prev_pos = torch.tensor([-100, -100])
            for i in range(len(episode_next_obs[0])):
                curr_pos = episode_next_obs[0][i][:2]
                if ((prev_pos-curr_pos)**2).sum() > 0.0003:
                    prev_pos = curr_pos
                    plt.plot(curr_pos[0], curr_pos[1], 'ro', markersize=15,
                             alpha=(curr_rew[i]-curr_rew.min())/(curr_rew.max()-curr_rew.min()))

            # plot behaviour & visualise belief in env
            # plot the trajectory
            plt.plot(episode_prev_obs[0][:, 0].cpu(), episode_prev_obs[0][:, 1].cpu(), '.')
            # plot the mountain
            angle = np.linspace(0, 2 * np.pi, 100)
            for r in [0.5, 0.4, 0.3, 0.2, 0.1]:
                plt.plot(r * np.cos(angle), r * np.sin(angle), 'r-', alpha=1-1.8*r)
            # plot the possible goals
            angle = np.linspace(0, 2 * np.pi, 100)
            plt.plot(np.cos(angle), np.sin(angle), 'k--', alpha=0.5)
            # plot the goal
            plt.plot(env.get_task()[0], env.get_task()[1], 'r*', markersize=10)
            # save figure that shows policy behaviour
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.tight_layout()
            plt.title('' + str(iter_idx) + ' ' + all_reward_names[b])
            if image_folder is not None:
                plt.savefig(f'{image_folder}/{iter_idx}_{all_reward_names[b]}')
                plt.close()
            else:
                plt.show()

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns
