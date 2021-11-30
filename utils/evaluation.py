import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             intrinsic_reward,
             encoder=None,
             num_episodes=None,
             vae=None,
             ):

    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minium of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)

    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    sparse_returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    dense_returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    returns_bonus_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # individual
    returns_bonus_belief_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    returns_bonus_state_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    returns_bonus_hyperstate_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    returns_bonus_vae_loss_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    successes = torch.zeros((num_processes, num_episodes + 1), dtype=torch.bool)

    # --- initialise environments and latents ---

    envs = make_vec_envs(seed=args.seed * 42 + iter_idx, num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms,
                         args=args)
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task = utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
        latent_sample = latent_sample.squeeze(0)
        latent_mean = latent_mean.squeeze(0)
        latent_logvar = latent_logvar.squeeze(0)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            prev_state = copy.copy(state)

            _, action = utl.select_action(args=args,
                                          policy=policy,
                                          state=state,
                                          belief=belief,
                                          task=task,
                                          latent_sample=latent_sample,
                                          latent_mean=latent_mean,
                                          latent_logvar=latent_logvar,
                                          deterministic=True)

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            done_mdp = [info['done_mdp'] for info in infos]
            # keep track of dense and sparse rewards
            sparse_rew = torch.tensor([info['sparse_reward'] for info in infos]).view(-1).to(device) if (
                        'sparse_reward' in infos[0]) else 0
            dense_rew = torch.tensor([info['dense_reward'] for info in infos]).view(-1).to(device) if (
                        'dense_reward' in infos[0]) else 0

            if encoder is not None:
                # update the hidden state
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                              next_obs=state.clone(),
                                                                                              action=action,
                                                                                              reward=rew_raw,
                                                                                              done=None,
                                                                                              hidden_state=hidden_state)
                # overwrite belief given by env with the latent/mean
                belief = torch.cat((latent_mean, latent_logvar), dim=1)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)
            sparse_returns_per_episode[range(num_processes), task_count] += sparse_rew
            dense_returns_per_episode[range(num_processes), task_count] += dense_rew

            # compute reward bonus for the current state/belief pair
            if args.add_exploration_bonus:
                rew_bonus, intrinsic_rew_state, \
                intrinsic_rew_belief, intrinsic_rew_hyperstate, \
                intrinsic_rew_vae_loss = intrinsic_reward.reward(state=state,
                                                                 belief=belief,
                                                                 return_individual=True,
                                                                 vae=vae,
                                                                 latent_mean=[latent_mean],
                                                                 latent_logvar=[latent_logvar],
                                                                 batch_prev_obs=prev_state.unsqueeze(0),
                                                                 batch_next_obs=state.unsqueeze(0),
                                                                 batch_actions=action.unsqueeze(0),
                                                                 batch_rewards=rew_raw.unsqueeze(0),
                                                                 batch_tasks=task.unsqueeze(0),
                                                                 )

                returns_bonus_per_episode[range(num_processes), task_count] += rew_bonus.view(-1)
                if args.exploration_bonus_state:
                    returns_bonus_state_per_episode[range(num_processes), task_count] += intrinsic_rew_state.view(-1)
                if args.exploration_bonus_belief:
                    returns_bonus_belief_per_episode[range(num_processes), task_count] += intrinsic_rew_belief.view(-1)
                if args.exploration_bonus_hyperstate:
                    returns_bonus_hyperstate_per_episode[
                        range(num_processes), task_count] += intrinsic_rew_hyperstate.view(-1)
                if args.exploration_bonus_vae_error and (vae is not None):
                    returns_bonus_vae_loss_per_episode[range(num_processes), task_count] += intrinsic_rew_vae_loss.view(
                        -1)

            # update success rates
            if 'success' in infos[0]:
                # |= does an in-place "or"
                successes[range(num_processes), task_count] |= torch.tensor([bool(info['success']) for info in infos])

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, done_indices, state)

    envs.close()

    return_list = [
        returns_per_episode[:, :num_episodes],
        sparse_returns_per_episode[:, :num_episodes],
        dense_returns_per_episode[:, :num_episodes],
        returns_bonus_per_episode[:, :num_episodes],
        returns_bonus_state_per_episode[:, :num_episodes],
        returns_bonus_belief_per_episode[:, :num_episodes],
        returns_bonus_hyperstate_per_episode[:, :num_episodes],
        returns_bonus_vae_loss_per_episode[:, :num_episodes],
        successes.float()[:, :num_episodes] if 'success' in infos[0] else None,
    ]

    return return_list


def visualise_behaviour(args,
                        policy,
                        image_folder,
                        iter_idx,
                        ret_rms,
                        encoder=None,
                        reward_decoder=None,
                        state_decoder=None,
                        task_decoder=None,
                        compute_rew_reconstruction_loss=None,
                        compute_task_reconstruction_loss=None,
                        compute_state_reconstruction_loss=None,
                        compute_kl_loss=None,
                        intrinsic_reward=None,
                        vae=None,
                        ):
    # initialise environment
    env = make_vec_envs(seed=args.seed * 42 + iter_idx,
                        num_processes=1,
                        gamma=args.policy_gamma,
                        device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms,
                        rank_offset=args.num_processes + 42,  # not sure if the tmp folders would otherwise clash
                        args=args)

    # get a sample rollout
    unwrapped_env = env.venv.unwrapped.envs[0]
    if hasattr(unwrapped_env, 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
    if hasattr(unwrapped_env, 'visualise_behaviour'):
        # if possible, get it from the env directly
        # (this might visualise other things in addition)
        traj = unwrapped_env.visualise_behaviour(env=env,
                                                 args=args,
                                                 policy=policy,
                                                 iter_idx=iter_idx,
                                                 encoder=encoder,
                                                 reward_decoder=reward_decoder,
                                                 state_decoder=state_decoder,
                                                 task_decoder=task_decoder,
                                                 image_folder=image_folder,
                                                 intrinsic_reward=intrinsic_reward,
                                                 vae=vae,
                                                 )
    else:
        traj = get_test_rollout(args, env, policy, encoder)

    if hasattr(unwrapped_env, 'visualise_reward_bonus'):
        unwrapped_env.visualise_reward_bonus(env=env,
                                             args=args,
                                             encoder=encoder,
                                             vae=vae,
                                             policy=policy,
                                             iter_idx=iter_idx,
                                             image_folder=image_folder,
                                             intrinsic_reward=intrinsic_reward,
                                             )

    # get the task that was used
    episode_task = torch.from_numpy(np.array(env.get_task())).to(device).float()

    latent_means, latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, episode_returns = traj

    if latent_means is not None:
        plot_latents(latent_means, latent_logvars,
                     image_folder=image_folder,
                     iter_idx=iter_idx
                     )

        if not args.disable_decoder:
            plot_vae_loss(latent_means,
                          latent_logvars,
                          episode_prev_obs,
                          episode_next_obs,
                          episode_actions,
                          episode_rewards,
                          episode_task,
                          image_folder=image_folder,
                          iter_idx=iter_idx,
                          reward_decoder=reward_decoder,
                          state_decoder=state_decoder,
                          task_decoder=task_decoder,
                          compute_task_reconstruction_loss=compute_task_reconstruction_loss,
                          compute_rew_reconstruction_loss=compute_rew_reconstruction_loss,
                          compute_state_reconstruction_loss=compute_state_reconstruction_loss,
                          compute_kl_loss=compute_kl_loss,
                          )

    env.close()


def get_test_rollout(args, env, policy, encoder=None):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    if encoder is not None:
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means = episode_latent_logvars = None

    # --- roll out policy ---

    # (re)set environment
    env.reset_task()
    state, belief, task = utl.reset_env(env, args)
    state = state.reshape((1, -1)).to(device)
    task = task.view(-1) if task is not None else None

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

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

        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(state.clone())

            latent = utl.get_latent_for_policy(args,
                                               latent_sample=curr_latent_sample,
                                               latent_mean=curr_latent_mean,
                                               latent_logvar=curr_latent_logvar)
            _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)
            action = action.reshape((1, -1))

            # observe reward and next obs
            (state, belief, task), (rew_raw, rew_normalised), done, infos = utl.env_step(env, action, args)
            state = state.reshape((1, -1)).to(device)
            task = task.view(-1) if task is not None else None

            if encoder is not None:
                # update task embedding
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                    action.reshape((1, -1)).float().to(device),
                    state,
                    rew_raw.reshape((1, 1)).float().to(device),
                    hidden_state,
                    return_prior=False)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]['done_mdp']:
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    # clean up
    if encoder is not None:
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    env.close()

    return episode_latent_means, episode_latent_logvars, \
           episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns


def plot_latents(latent_means,
                 latent_logvars,
                 image_folder,
                 iter_idx,
                 ):
    """
    Plot mean/variance over time
    """

    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])

    latent_means = torch.cat(latent_means).cpu().detach().numpy()
    latent_logvars = torch.cat(latent_logvars).cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(latent_means.shape[0]), latent_means, '-', alpha=0.5)
    plt.plot(range(latent_means.shape[0]), latent_means.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_means.max() - latent_means.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_means.min() - span * 0.05, latent_means.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent mean', fontsize=15)

    plt.subplot(1, 2, 2)
    latent_var = np.exp(latent_logvars)
    plt.plot(range(latent_logvars.shape[0]), latent_var, '-', alpha=0.5)
    plt.plot(range(latent_logvars.shape[0]), latent_var.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_var.max() - latent_var.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_var.min() - span * 0.05, latent_var.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent variance', fontsize=15)

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_latents'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_vae_loss(latent_means,
                  latent_logvars,
                  prev_obs,
                  next_obs,
                  actions,
                  rewards,
                  task,
                  image_folder,
                  iter_idx,
                  reward_decoder,
                  state_decoder,
                  task_decoder,
                  compute_task_reconstruction_loss,
                  compute_rew_reconstruction_loss,
                  compute_state_reconstruction_loss,
                  compute_kl_loss
                  ):
    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])
    num_samples = 10  # how many samples to use to get an average/std ELBO loss

    latent_means = torch.cat(latent_means)
    latent_logvars = torch.cat(latent_logvars)

    prev_obs = torch.cat(prev_obs).to(device)
    next_obs = torch.cat(next_obs).to(device)
    actions = torch.cat(actions).to(device)
    rewards = torch.cat(rewards).to(device)

    # - we will try to make predictions for each tuple in trajectory, hence we need to expand the targets
    prev_obs = prev_obs.unsqueeze(0).expand(num_samples, *prev_obs.shape).to(device)
    next_obs = next_obs.unsqueeze(0).expand(num_samples, *next_obs.shape).to(device)
    actions = actions.unsqueeze(0).expand(num_samples, *actions.shape).to(device)
    rewards = rewards.unsqueeze(0).expand(num_samples, *rewards.shape).to(device)

    rew_reconstr_mean = []
    rew_reconstr_std = []
    rew_pred_std = []

    state_reconstr_mean = []
    state_reconstr_std = []
    state_pred_std = []

    task_reconstr_mean = []
    task_reconstr_std = []
    task_pred_std = []

    # compute the sum of ELBO_t's by looping through (trajectory length + prior)
    for i in range(len(latent_means)):

        curr_latent_mean = latent_means[i]
        curr_latent_logvar = latent_logvars[i]

        # compute the reconstruction loss
        latent_samples = utl.sample_gaussian(curr_latent_mean.view(-1), curr_latent_logvar.view(-1), num_samples)

        # expand: each latent sample will be used to make predictions for the entire trajectory
        len_traj = prev_obs.shape[1]

        # compute reconstruction losses
        if task_decoder is not None:
            loss_task, task_pred = compute_task_reconstruction_loss(latent_samples, task, return_predictions=True)

            # average/std across the different samples
            task_reconstr_mean.append(loss_task.mean())
            task_reconstr_std.append(loss_task.std())
            task_pred_std.append(task_pred.std())

        latent_samples = latent_samples.unsqueeze(1).expand(num_samples, len_traj, latent_samples.shape[-1])

        if reward_decoder is not None:
            loss_rew, rew_pred = compute_rew_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                 actions, rewards, return_predictions=True)
            # sum along length of trajectory
            loss_rew = loss_rew.sum(dim=1)
            rew_pred = rew_pred.sum(dim=1)

            # average/std across the different samples
            rew_reconstr_mean.append(loss_rew.mean())
            rew_reconstr_std.append(loss_rew.std())
            rew_pred_std.append(rew_pred.std())

        if state_decoder is not None:
            loss_state, state_pred = compute_state_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                       actions, return_predictions=True)
            # sum along length of trajectory
            loss_state = loss_state.sum(dim=1)
            state_pred = state_pred.sum(dim=1)

            # average/std across the different samples
            state_reconstr_mean.append(loss_state.mean())
            state_reconstr_std.append(loss_state.std())
            state_pred_std.append(state_pred.std())

    # kl term
    vae_kl_term = compute_kl_loss(latent_means, latent_logvars, None)

    # --- plot KL term ---

    x = range(len(vae_kl_term))
    vae_kl_term = vae_kl_term.cpu().detach().numpy()

    plt.plot(x, vae_kl_term, 'b-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = vae_kl_term.max() - vae_kl_term.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [vae_kl_term.min() - span * 0.05, vae_kl_term.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('KL term', fontsize=15)
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_kl'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    # --- plot rew reconstruction ---

    if reward_decoder is not None:

        rew_reconstr_mean = torch.stack(rew_reconstr_mean).detach().cpu().numpy()
        rew_reconstr_std = torch.stack(rew_reconstr_std).detach().cpu().numpy()
        rew_pred_std = torch.stack(rew_pred_std).detach().cpu().numpy()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        p = plt.plot(x, rew_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               rew_reconstr_mean - rew_reconstr_std,
                               rew_reconstr_mean + rew_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (rew_reconstr_mean - rew_reconstr_std).min()
            max_y = (rew_reconstr_mean + rew_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('reward reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, rew_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = rew_pred_std.max() - rew_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [rew_pred_std.min() - span * 0.05, rew_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of rew reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_rew_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

    # --- plot state reconstruction ---

    if state_decoder is not None:

        plt.figure(figsize=(12, 5))

        state_reconstr_mean = torch.stack(state_reconstr_mean).detach().cpu().numpy()
        state_reconstr_std = torch.stack(state_reconstr_std).detach().cpu().numpy()
        state_pred_std = torch.stack(state_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, state_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               state_reconstr_mean - state_reconstr_std,
                               state_reconstr_mean + state_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (state_reconstr_mean - state_reconstr_std).min()
            max_y = (state_reconstr_mean + state_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('state reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, state_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = state_pred_std.max() - state_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [state_pred_std.min() - span * 0.05, state_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of state reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_state_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

    # --- plot task reconstruction ---

    if task_decoder is not None:

        plt.figure(figsize=(12, 5))

        task_reconstr_mean = torch.stack(task_reconstr_mean).detach().cpu().numpy()
        task_reconstr_std = torch.stack(task_reconstr_std).detach().cpu().numpy()
        task_pred_std = torch.stack(task_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, task_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               task_reconstr_mean - task_reconstr_std,
                               task_reconstr_mean + task_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (task_reconstr_mean - task_reconstr_std).min()
            max_y = (task_reconstr_mean + task_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('task reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, task_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = task_pred_std.max() - task_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [task_pred_std.min() - span * 0.05, task_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of task reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_task_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()
