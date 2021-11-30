import os
import time

import gym
import numpy as np
import torch

from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from exploration.exploration_bonus import ExplorationBonus
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """ Meta-Learner class with the main training loop. """

    def __init__(self, args):

        self.args = args
        utl.set_seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments
        self.envs = make_vec_envs(seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  args=args
                                  )

        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = self.envs._max_episode_steps
        args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise VAE
        self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)

        # initialise rew bonus
        if self.args.add_exploration_bonus:
            self.intrinsic_reward = ExplorationBonus(args=self.args,
                                                     logger=self.logger,
                                                     dim_state=self.args.state_dim,
                                                     encoder=self.vae.encoder,
                                                     rollout_storage=self.vae.rollout_storage,
                                                     )
        else:
            self.intrinsic_reward = None

        # initialise policy
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        return OnlineStorage(
            args=self.args,
            num_steps=self.args.policy_num_steps,
            num_processes=self.args.num_processes,
            state_dim=self.args.state_dim,
            latent_dim=self.args.latent_dim,
            belief_dim=self.args.belief_dim,
            task_dim=self.args.task_dim,
            action_space=self.args.action_space,
            hidden_size=self.args.encoder_gru_hidden_size,
            normalise_rewards=self.args.norm_rew_for_policy,
            add_exploration_bonus=self.args.add_exploration_bonus,
            intrinsic_reward=self.intrinsic_reward,
        )

    def initialise_policy(self):

        policy_net = Policy(
            args=self.args,
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            hidden_layers=self.args.policy_layers,
            action_space=self.envs.action_space,
        ).to(device)

        policy = PPO(
            args=self.args,
            actor_critic=policy_net,
            entropy_coef=self.args.policy_entropy_coef,
            lr=self.args.lr_policy,
            policy_anneal_lr=self.args.policy_anneal_lr,
            train_steps=self.num_updates,
            num_epochs=self.args.ppo_num_epochs,
            num_mini_batches=self.args.ppo_num_minibatch,
            clip_param=self.args.ppo_clip_param,
            optimiser_vae=self.vae.optimiser_vae,
        )

        return policy

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        # reset environments
        prev_state, belief, task = utl.reset_env(self.envs, self.args)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)
        self.policy_storage.tasks[0].copy_(task)
        if belief is not None:
            self.policy_storage.beliefs[0].copy_(belief)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)

        intrinsic_reward_is_pretrained = False
        for self.iter_idx in range(self.num_updates):

            # first, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory()

            # add this initial hidden state to the policy storage
            assert len(self.policy_storage.latent_mean) == 0  # make sure we emptied buffers
            self.policy_storage.hidden_states[0].copy_(hidden_state)
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())

            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                # take step in the environment
                [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action,
                                                                                                  self.args)

                done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                        encoder=self.vae.encoder,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                if not self.args.disable_decoder:
                    self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    task.clone())

                # add new observations to intrinsic reward
                if self.args.add_exploration_bonus:
                    beliefs = torch.cat((latent_mean, latent_logvar), dim=-1)
                    self.intrinsic_reward.add(next_state, beliefs, action.detach())

                # add the obs before reset to the policy storage
                # (necessary for rew bonus)
                self.policy_storage.next_state[step] = next_state.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, belief, task = utl.reset_env(self.envs, self.args, done_indices, next_state)

                # add experience to policy buffer
                self.policy_storage.insert(
                    state=next_state,
                    belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done,
                    hidden_states=hidden_state.squeeze(0),
                    latent_sample=latent_sample,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar,
                )

                prev_state = next_state

                self.frames += self.args.num_processes

            # --- UPDATE ---

            if (len(self.vae.rollout_storage) == 0 and not self.args.size_vae_buffer == 0) or \
                    (self.args.precollect_len > self.frames):
                print('Not updating yet because; filling up the VAE buffer.')
                self.policy_storage.after_update()
                continue

            # pretrain RND model once to bring it on right scale
            if self.args.add_exploration_bonus and not intrinsic_reward_is_pretrained:
                # compute returns once - this will normalise the RND inputs!
                next_value = self.get_value(state=next_state,
                                            belief=belief,
                                            task=task,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar)
                self.policy_storage.compute_returns(next_value, self.args.policy_gamma,
                                                    self.args.policy_tau,
                                                    use_proper_time_limits=self.args.use_proper_time_limits,
                                                    vae=self.vae)
                self.intrinsic_reward.update(self.args.num_frames, self.iter_idx,
                                             log=False)  # (calling with max num of frames to init all networks)
                intrinsic_reward_is_pretrained = True
            else:

                train_stats = self.update(state=prev_state,
                                          belief=belief,
                                          task=task,
                                          latent_sample=latent_sample,
                                          latent_mean=latent_mean,
                                          latent_logvar=latent_logvar)

                # log
                run_stats = [action, self.policy_storage.action_log_probs, value]
                if train_stats is not None:
                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)

                # update intrinsic reward model
                if self.args.add_exploration_bonus:
                    if self.iter_idx % self.args.rnd_update_frequency == 0:
                        self.intrinsic_reward.update(self.frames, self.iter_idx)

            # clean up after update
            self.policy_storage.after_update()

        self.envs.close()

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(
            actions=act,
            states=next_obs,
            rewards=rew,
            hidden_state=None,
            return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        latent = utl.get_latent_for_policy(self.args, latent_sample=latent_sample, latent_mean=latent_mean,
                                           latent_logvar=latent_logvar)
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()

    def update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # update policy (if we have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar)

            # compute returns for current rollouts
            self.policy_storage.compute_returns(next_value,
                                                self.args.policy_gamma, self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits,
                                                vae=self.vae
                                                )

            # update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time):

        # --- visualise behaviour of policy ---

        if (self.iter_idx + 1) % self.args.vis_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         encoder=self.vae.encoder,
                                         reward_decoder=self.vae.reward_decoder,
                                         state_decoder=self.vae.state_decoder,
                                         task_decoder=self.vae.task_decoder,
                                         compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
                                         compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
                                         compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
                                         compute_kl_loss=self.vae.compute_kl_loss,
                                         intrinsic_reward=self.intrinsic_reward,
                                         vae=self.vae,
                                         )

        # --- evaluate policy ----

        if (self.iter_idx + 1) % self.args.eval_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    intrinsic_reward=self.intrinsic_reward,
                                                    iter_idx=self.iter_idx,
                                                    num_episodes=None,
                                                    encoder=self.vae.encoder,
                                                    vae=self.vae,
                                                    )

            (returns_per_episode, sparse_returns_per_episode, dense_returns_per_episode,
             returns_bonus_per_episode,
             returns_bonus_state_per_episode, returns_bonus_belief_per_episode,
             returns_bonus_hyperstate_per_episode,
             returns_bonus_vae_loss_per_episode,
             success_per_episode) = returns_per_episode

            # get the average across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            if success_per_episode is not None:
                successes_avg = success_per_episode.mean(dim=0)
            sparse_returns_avg = sparse_returns_per_episode.mean(dim=0)
            dense_returns_avg = dense_returns_per_episode.mean(dim=0)
            returns_bonus_avg = returns_bonus_per_episode.mean(dim=0)
            returns_bonus_state_avg = returns_bonus_state_per_episode.mean(dim=0)
            returns_bonus_belief_avg = returns_bonus_belief_per_episode.mean(dim=0)
            returns_bonus_hyperstate_avg = returns_bonus_hyperstate_per_episode.mean(dim=0)
            returns_bonus_vae_loss_avg = returns_bonus_vae_loss_per_episode.mean(dim=0)

            for k in range(len(returns_avg)):
                # avg
                self.logger.add(f'return_avg_per_iter/episode_{k + 1}', returns_avg[k], self.iter_idx)
                self.logger.add(f'return_avg_per_frame/episode_{k + 1}', returns_avg[k], self.frames)
                if success_per_episode is not None:
                    self.logger.add(f'success_avg_per_iter/episode_{k + 1}', successes_avg[k], self.iter_idx)
                    self.logger.add(f'success_avg_per_frame/episode_{k + 1}', successes_avg[k], self.frames)
                # sparse
                self.logger.add(f'sparse_return_avg_per_iter/episode_{k + 1}', sparse_returns_avg[k],
                                self.iter_idx)
                self.logger.add(f'sparse_return_avg_per_frame/episode_{k + 1}', sparse_returns_avg[k],
                                self.frames)
                # dense
                self.logger.add(f'dense_return_avg_per_iter/episode_{k + 1}', dense_returns_avg[k],
                                self.iter_idx)
                self.logger.add(f'dense_return_avg_per_frame/episode_{k + 1}', dense_returns_avg[k],
                                self.frames)
                # avg bonus
                self.logger.add(f'return_bonus_avg_per_iter/episode_{k + 1}',
                                returns_bonus_avg[k], self.iter_idx)
                self.logger.add(f'return_bonus_avg_per_frame/episode_{k + 1}',
                                returns_bonus_avg[k], self.frames)
                if self.args.add_exploration_bonus:
                    # individual bonuses: states
                    if self.args.exploration_bonus_state:
                        self.logger.add(f'return_bonus_state_avg_per_iter/episode_{k + 1}',
                                        returns_bonus_state_avg[k], self.iter_idx)
                        self.logger.add(f'return_bonus_state_avg_per_frame/episode_{k + 1}',
                                        returns_bonus_state_avg[k], self.frames)
                    # individual bonuses: belief
                    if self.args.exploration_bonus_belief:
                        self.logger.add(f'return_bonus_belief_avg_per_iter/episode_{k + 1}',
                                        returns_bonus_belief_avg[k], self.iter_idx)
                        self.logger.add(f'return_bonus_belief_avg_per_frame/episode_{k + 1}',
                                        returns_bonus_belief_avg[k], self.frames)
                    # individual bonuses: hyperstates
                    if self.args.exploration_bonus_hyperstate:
                        self.logger.add(f'return_bonus_hyperstate_avg_per_iter/episode_{k + 1}',
                                        returns_bonus_hyperstate_avg[k], self.iter_idx)
                        self.logger.add(f'return_bonus_hyperstate_avg_per_frame/episode_{k + 1}',
                                        returns_bonus_hyperstate_avg[k], self.frames)
                    # individual bonuses: vae loss
                    if hasattr(self.args, 'exploration_bonus_vae_error') and self.args.exploration_bonus_vae_error:
                        self.logger.add(f'return_bonus_vae_loss_avg_per_iter/episode_{k + 1}',
                                        returns_bonus_vae_loss_avg[k], self.iter_idx)
                        self.logger.add(f'return_bonus_vae_loss_avg_per_frame/episode_{k + 1}',
                                        returns_bonus_vae_loss_avg[k], self.frames)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (train): {returns_avg[-1].item()} \n"
                  )

        # --- save models ---

        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy.pt"))
            torch.save(self.vae.encoder, os.path.join(save_path, f"encoder.pt"))
            if self.vae.state_decoder is not None:
                torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder.pt"))
            if self.vae.reward_decoder is not None:
                torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder.pt"))
            if self.vae.task_decoder is not None:
                torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder.pt"))

            # save normalisation params of envs
            if self.args.norm_rew_for_policy:
                rew_rms = self.envs.venv.ret_rms
                utl.save_obj(rew_rms, save_path, f"env_rew_rms")

        # --- log some other things ---

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        try:
                            param_grad_mean = np.mean(
                                [param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                            self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
                        except:
                            pass
