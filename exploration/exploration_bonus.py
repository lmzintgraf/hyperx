import numpy as np
import torch

from exploration.rnd.rnd_bonus import RNDRewardBonus
from exploration.rollout_storage import RolloutStorage
from utils import helpers as utl
from utils.helpers import RunningMeanStd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExplorationBonus(object):
    def __init__(self,
                 args,
                 logger,
                 dim_state,
                 encoder=None,
                 rollout_storage=None
                 ):
        """ HyperX Exploration Bonuses. """

        self.args = args
        self.logger = logger
        self.dim_state = dim_state
        self.dim_action = args.action_dim
        self.encoder = encoder
        self.anneal_weights = self.args.anneal_exploration_bonus_weights
        self.curr_weight = 1.0

        # check if we are in a BAMDP and get size of belief
        if self.args.pass_belief_to_policy:
            self.dim_belief = self.args.belief_dim
        elif self.args.pass_latent_to_policy:
            self.dim_belief = self.args.latent_dim * 2
        else:
            self.dim_belief = 0

        # if we have specified which parts of the state space to do exploration on,
        # overwrite the state dimensionality with the smaller one
        if self.args.state_expl_idx is None:
            self.state_expl_idx = torch.tensor([i for i in range(self.dim_state)]).to(device)
        else:
            self.state_expl_idx = torch.tensor(self.args.state_expl_idx).to(device)

        # Initialise a rollout storage if necessary for the methods
        if rollout_storage is not None:
            self.rollout_storage = rollout_storage
        else:
            self.rollout_storage = self.initialise_storage()

        # correct state dimension if we only use the rew bonus on a subspace
        if len(self.state_expl_idx) > 0:
            self.dim_state = len(self.state_expl_idx)

        # Make different types of reward bonuses.
        self.intrinsic_rew_hyperstate = None
        self.intrinsic_rew_state = None
        self.intrinsic_rew_belief = None
        # - hyperstate
        if self.args.exploration_bonus_hyperstate:
            dim_inputs = [self.dim_state, self.dim_belief]
            self.intrinsic_rew_hyperstate = RNDRewardBonus(args, logger, dim_inputs, self.rollout_storage)
        # - state
        if self.args.exploration_bonus_state:
            dim_inputs = [self.dim_state]
            self.intrinsic_rew_state = RNDRewardBonus(args, logger, dim_inputs, self.rollout_storage)
        # - belief
        if self.args.exploration_bonus_belief:
            dim_inputs = [self.dim_belief]
            self.intrinsic_rew_belief = RNDRewardBonus(args, logger, dim_inputs, self.rollout_storage)

        # for normalising the intrinsic rewards
        self.epsilon = 1e-8
        self.cliprew = args.intrinsic_rew_clip_rewards
        self.gamma = args.policy_gamma
        if self.args.exploration_bonus_hyperstate:
            # for normalising the rew
            self.ret_hs_rms = RunningMeanStd(shape=())
            # discounted return for each environment (the buffer is to wait until all envs are done)
            self.ret_hs = torch.zeros(size=(self.args.num_processes, 1)).to(device)
        if self.args.exploration_bonus_state:
            # for normalising the rew
            self.ret_s_rms = RunningMeanStd(shape=())
            # discounted return for each environment
            self.ret_s = torch.zeros(size=(self.args.num_processes, 1)).to(device)
        if self.args.exploration_bonus_belief:
            # for normalising the rew
            self.ret_b_rms = RunningMeanStd(shape=())
            # discounted return for each environment
            self.ret_b = torch.zeros(size=(self.args.num_processes, 1)).to(device)
        if self.args.exploration_bonus_vae_error:
            # for normalising the rew
            self.ret_v_rms = RunningMeanStd(shape=())
            # discounted return for each environment
            self.ret_v = torch.zeros(size=(self.args.num_processes, 1)).to(device)

    def initialise_storage(self):

        # initialise rollout storage for all experience
        return RolloutStorage(max_buffer_size=self.args.rnd_buffer_size,
                              env_state_shape=[self.dim_state],
                              action_shape=[self.dim_action],
                              belief_shape=[self.dim_belief])

    def add(self, states, beliefs, actions):
        # if we're using our own rolout storage (instead of a shared one with the VAE), add data
        if isinstance(self.rollout_storage, RolloutStorage):
            self.rollout_storage.insert(states, beliefs, actions)

    def reward(self,
               state,
               belief,
               return_individual=False,
               normalise=True,
               update_normalisation=False,
               done=None,
               # for the vae loss
               vae=None,
               latent_mean=None,
               latent_logvar=None,
               batch_prev_obs=None,
               batch_next_obs=None,
               batch_actions=None,
               batch_rewards=None,
               batch_tasks=None,
               ):
        """
        Returns an intrinsic reward.
        Make sure that state and latent_mean/latent_logvar are properly aligned!
        We typically want to use s^+_{t+1} for t=0,...,H-1, i.e., skip the first observation and the prior.
        """

        if self.args.exploration_bonus_vae_error:
            assert vae is not None

        with torch.no_grad():

            # select part of state space that we want to explore
            if 0 < len(self.state_expl_idx) < state.shape[-1]:
                state = state.to(device)
                state = state.index_select(dim=-1, index=self.state_expl_idx)
            else:
                state = state.clone()

            # Compute bonus for the Hyper-State
            intrinsic_rew_hyperstate = 0
            if self.args.exploration_bonus_hyperstate:
                assert belief.shape[0] == state.shape[0]
                inputs = [state, belief]
                intrinsic_rew_hyperstate = self.intrinsic_rew_hyperstate.reward(
                    inputs, update_normalisation=update_normalisation)
                if update_normalisation and intrinsic_rew_hyperstate.shape[1] > 1:
                    for i in range(intrinsic_rew_hyperstate.shape[0]):
                        self.ret_hs = self.ret_hs * self.gamma + intrinsic_rew_hyperstate[i]
                        self.ret_hs_rms.update(self.ret_hs)
                        self.ret_hs[done[i] == 1] = 0.
                if normalise:
                    intrinsic_rew_hyperstate = intrinsic_rew_hyperstate / torch.sqrt(self.ret_hs_rms.var + self.epsilon)
                    if self.cliprew is not None:
                        intrinsic_rew_hyperstate = torch.clamp(intrinsic_rew_hyperstate, -self.cliprew, self.cliprew)
                intrinsic_rew_hyperstate *= self.args.weight_exploration_bonus_hyperstate

            # Compute bonus for the state
            intrinsic_rew_state = 0
            if self.args.exploration_bonus_state:
                inputs = [state]
                intrinsic_rew_state = self.intrinsic_rew_state.reward(inputs, update_normalisation=update_normalisation)
                if normalise:
                    if update_normalisation and intrinsic_rew_state.shape[1] > 1:
                        for i in range(intrinsic_rew_state.shape[0]):
                            self.ret_s = self.ret_s * self.gamma + intrinsic_rew_state[i]
                            self.ret_s_rms.update(self.ret_s)
                            self.ret_s[done[i] == 1] = 0.
                    intrinsic_rew_state = intrinsic_rew_state / torch.sqrt(self.ret_s_rms.var + self.epsilon)
                    if self.cliprew is not None:
                        intrinsic_rew_state = torch.clamp(intrinsic_rew_state, -self.cliprew, self.cliprew)
                intrinsic_rew_state *= self.args.weight_exploration_bonus_state

            # Compute bonus for the belief
            intrinsic_rew_belief = 0
            if self.args.exploration_bonus_belief:
                assert belief.shape[0] == state.shape[0]
                inputs = [belief]
                intrinsic_rew_belief = self.intrinsic_rew_belief.reward(
                    inputs, update_normalisation=update_normalisation)
                if normalise:
                    if update_normalisation and intrinsic_rew_belief.shape[1] > 1:
                        for i in range(intrinsic_rew_belief.shape[0]):
                            self.ret_b = self.ret_b * self.gamma + intrinsic_rew_belief[i]
                            self.ret_b_rms.update(self.ret_b)
                            self.ret_b[done[i] == 1] = 0.
                    intrinsic_rew_belief = intrinsic_rew_belief / torch.sqrt(self.ret_b_rms.var + self.epsilon)
                    if self.cliprew is not None:
                        intrinsic_rew_belief = torch.clamp(intrinsic_rew_belief, -self.cliprew, self.cliprew)
                intrinsic_rew_belief *= self.args.weight_exploration_bonus_belief

            # compute the bonus from the vae loss
            intrinsic_vae_bonus = 0
            if self.args.exploration_bonus_vae_error:

                # take one sample for each ELBO term
                if latent_mean[0].shape[0] == batch_prev_obs.shape[1]:  # batchsize on dim 1
                    latent_mean = torch.stack(latent_mean)
                    latent_logvar = torch.stack(latent_logvar)
                else:
                    latent_mean = torch.stack(latent_mean, dim=0)
                    latent_logvar = torch.stack(latent_logvar, dim=0)
                latent_samples = utl.sample_gaussian(latent_mean, latent_logvar)
                if self.args.decode_reward:
                    # given: current belief b_t (which includes r_t), predict reward r_t
                    rew_reconstruction_loss = vae.compute_rew_reconstruction_loss(
                        latent_samples[1:] if latent_samples.shape[0] > 1 else latent_samples,
                        batch_prev_obs,
                        batch_next_obs,
                        batch_actions,
                        batch_rewards).unsqueeze(-1)
                else:
                    rew_reconstruction_loss = 0
                if self.args.decode_state:
                    # given: current belief b_t (which includes r_t, s_{t+1}), predict reward s_{t+1}
                    state_reconstruction_loss = vae.compute_state_reconstruction_loss(
                        latent_samples[1:] if latent_samples.shape[0] > 1 else latent_samples,
                        batch_prev_obs,
                        batch_next_obs,
                        batch_actions).unsqueeze(-1)
                else:
                    state_reconstruction_loss = 0
                if self.args.decode_task:
                    # given: current belief b_t (which includes r_t, s_{t+1}), predict reward s_{t+1}
                    task_reconstruction_loss = vae.compute_task_reconstruction_loss(
                        latent_samples[1:] if latent_samples.shape[0] > 1 else latent_samples,
                        batch_tasks).unsqueeze(-1)
                else:
                    task_reconstruction_loss = 0
                # average across dimensions (the above returns mse tuple-wise)
                intrinsic_vae_bonus = self.args.rew_loss_coeff * rew_reconstruction_loss + \
                                      self.args.state_loss_coeff * state_reconstruction_loss + \
                                      self.args.task_loss_coeff * task_reconstruction_loss
                if intrinsic_vae_bonus.shape[0] == 1:
                    intrinsic_vae_bonus = intrinsic_vae_bonus.squeeze(0)
                if normalise:
                    if update_normalisation and intrinsic_vae_bonus.shape[1] > 1:
                        for i in range(intrinsic_vae_bonus.shape[0]):
                            self.ret_v = self.ret_v * self.gamma + intrinsic_vae_bonus[i]
                            self.ret_v_rms.update(self.ret_v)
                            self.ret_v[done[i] == 1] = 0.
                    intrinsic_vae_bonus = intrinsic_vae_bonus / torch.sqrt(self.ret_v_rms.var + self.epsilon)
                    if self.cliprew is not None:
                        intrinsic_vae_bonus = torch.clamp(intrinsic_vae_bonus, -self.cliprew, self.cliprew)

                intrinsic_vae_bonus *= self.args.weight_exploration_bonus_vae_error

            rew_bonus = intrinsic_rew_hyperstate + \
                        intrinsic_rew_state + \
                        intrinsic_rew_belief + \
                        intrinsic_vae_bonus

            if self.anneal_weights:
                rew_bonus *= self.curr_weight

            if not return_individual:
                return rew_bonus
            else:
                return rew_bonus, intrinsic_rew_state, intrinsic_rew_belief, intrinsic_rew_hyperstate, \
                       intrinsic_vae_bonus

    def update(self, frames, iter_idx, log=True):
        """ Updates the intrinsic reward model. """

        # anneal the weight linearly over the course of training
        if self.anneal_weights:
            self.curr_weight = 1 - frames / self.args.num_frames

        if self.encoder is not None:

            # get a batch of data (using s_{t+1}, i.e., skipping the first observation)
            num_enc_len = int(np.sqrt(self.args.rnd_batch_size))
            batchsize = self.args.rnd_batch_size // num_enc_len
            vae_prev_obs, vae_next_obs, vae_actions, \
            vae_rewards, vae_tasks, trajectory_lens = self.rollout_storage.get_batch(batchsize=batchsize)

            # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations
            # len_encoder will be of size:  number of trajectories x data_per_rollout

            # pass through encoder (outputs: max_traj_len x number of rollouts x latent_dim -- skipping the prior!)
            with torch.no_grad():
                _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                                states=vae_next_obs,
                                                                rewards=vae_rewards,
                                                                hidden_state=None,
                                                                return_prior=True)

            indices = [np.random.choice(range(0, t), num_enc_len) for t in trajectory_lens]
            vae_next_obs = torch.cat([vae_next_obs[indices[i], i, :] for i in range(len(indices))])
            latent_mean = torch.cat([latent_mean[indices[i] + 1, i, :] for i in range(len(indices))])
            latent_logvar = torch.cat([latent_logvar[indices[i] + 1, i, :] for i in range(len(indices))])

            state = vae_next_obs
            belief = torch.cat((latent_mean, latent_logvar), dim=1).detach()

        elif self.args.pass_belief_to_policy:
            # get a batch of data
            state, belief, actions = self.rollout_storage.get_batch(batchsize=self.args.rnd_batch_size)
            state = state.to(device)
            belief = belief.to(device)

        else:
            raise ValueError

        # select part of state space that we want to explore
        if 0 < len(self.state_expl_idx) < state.shape[-1]:
            state = state.index_select(dim=-1, index=self.state_expl_idx)
        else:
            state = state.clone()

        state = state.to(device)
        if belief is not None:
            belief = belief.to(device)

        # Compute bonus for the Hyper-State
        if self.args.exploration_bonus_hyperstate:
            inputs = [state, belief]
            loss_hyperstate = self.intrinsic_rew_hyperstate.update(inputs)
            if log and iter_idx > 1:
                self.logger.add('rnd/loss_hyperstate', loss_hyperstate.detach().cpu().numpy(), frames)

        # Compute bonus for the state
        if self.args.exploration_bonus_state:
            inputs = [state]
            loss_state = self.intrinsic_rew_state.update(inputs)
            if log and iter_idx > 1:
                self.logger.add('rnd/loss_state', loss_state.detach().cpu().numpy(), frames)

        # Compute bonus for the belief
        if self.args.exploration_bonus_belief:
            inputs = [belief]
            loss_belief = self.intrinsic_rew_belief.update(inputs)
            if log and iter_idx > 1:
                self.logger.add('rnd/loss_belief', loss_belief.detach().cpu().numpy(), frames)
