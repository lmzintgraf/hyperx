import os
import pickle
import random
import warnings
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if indices is not None:
        assert not isinstance(indices[0], bool)
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().float().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device)

    return state, belief, task


def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action


def env_step(env, action, args):
    act = squash_action(action.detach(), args)
    next_obs, reward, done, infos = env.step(act)

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device)

    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  belief=None,
                  task=None,
                  latent_sample=None, latent_mean=None, latent_logvar=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_sample=latent_sample, latent_mean=latent_mean,
                                   latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None
    action = action.to(device)
    return value, action


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):
    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent


def update_encoding(encoder, next_obs, action, reward, done, hidden_state):
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    # case: no resets
    if (done is None) or ((done == 0).sum() == done.shape[0]):
        with torch.no_grad():
            latent_sample, latent_mean, latent_logvar, hidden_state = encoder(actions=action.float(),
                                                                              states=next_obs,
                                                                              rewards=reward,
                                                                              hidden_state=hidden_state,
                                                                              return_prior=False)

    # case: all resets
    elif (done == 1).sum() == done.shape[0]:
        with torch.no_grad():
            latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(action.shape[0])
            latent_sample = latent_sample.squeeze(0)
            latent_mean = latent_mean.squeeze(0)
            latent_logvar = latent_logvar.squeeze(0)

    # case: some resets
    else:
        raise NotImplementedError

    return latent_sample, latent_mean, latent_logvar, hidden_state


def set_seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')


def recompute_embeddings(
        policy_storage,
        encoder,
        sample,
        update_idx,
        detach_every
):
    # get the prior
    latent_sample = [policy_storage.latent_samples[0].detach().clone()]
    latent_mean = [policy_storage.latent_mean[0].detach().clone()]
    latent_logvar = [policy_storage.latent_logvar[0].detach().clone()]

    latent_sample[0].requires_grad = True
    latent_mean[0].requires_grad = True
    latent_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        h = encoder.reset_hidden(h, policy_storage.done[i + 1])

        if policy_storage.done[i + 1].sum() > 0:
            ts, tm, tl, h = encoder.prior(h.shape[1])
            ts = ts.squeeze(0)
            tm = tm.squeeze(0)
            tl = tl.squeeze(0)
        else:
            ts, tm, tl, h = encoder(policy_storage.actions.float()[i:i + 1],
                                    policy_storage.next_state[i:i + 1],
                                    policy_storage.rewards_raw[i:i + 1],
                                    h,
                                    sample=sample,
                                    return_prior=False,
                                    detach_every=detach_every
                                    )

        latent_sample.append(ts)
        latent_mean.append(tm)
        latent_logvar.append(tl)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.latent_mean) - torch.cat(latent_mean)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) - torch.cat(latent_logvar)).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = latent_sample
    policy_storage.latent_mean = latent_mean
    policy_storage.latent_logvar = latent_logvar


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    if num is None:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        std = torch.exp(0.5 * logvar).repeat(num, 1)
        eps = torch.randn_like(std)
        mu = mu.repeat(num, 1)
        return eps.mul(std).add_(mu)


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def make_env(args, **kwargs):
    env_id = args.env_name
    env_type = 'gym'
    env = gym.make(env_id, **kwargs)
    return env, env_type


def get_task_dim(args):
    env, _ = make_env(args)
    return env.task_dim


def get_num_tasks(args):
    env = make_env(args)
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        print("Asking for number of tasks failed. Setting num_tasks=None.")
        num_tasks = None
    return num_tasks
