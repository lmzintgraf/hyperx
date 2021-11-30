"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import copy

import numpy as np
import torch

import utils.helpers as utl
from environments.env_utils.vec_env import VecEnvWrapper
from environments.env_utils.vec_env.dummy_vec_env import DummyVecEnv
from environments.env_utils.vec_env.subproc_vec_env import SubprocVecEnv
from environments.env_utils.vec_env.vec_normalize import VecNormalize
from environments.wrappers import TimeLimitMask
from environments.wrappers import VariBadWrapper


def make_env(seed, rank, episodes_per_task, args, **kwargs):
    def _thunk():

        env, env_type = utl.make_env(args, **kwargs)

        if seed is not None:
            env.seed(seed + rank)
            env.np_random = np.random.RandomState(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        env = VariBadWrapper(env=env, episodes_per_task=episodes_per_task,
                             env_type=env_type, add_done_flag=args.max_rollouts_per_task > 1)

        return env

    return _thunk


def make_vec_envs(seed, num_processes, gamma,
                  device, episodes_per_task,
                  normalise_rew, ret_rms,
                  args, rank_offset=0,
                  **kwargs):
    """
    :param ret_rms: running return and std for rewards
    """
    envs = [make_env(seed=seed, rank=rank_offset + i,
                     episodes_per_task=episodes_per_task,
                     args=args, **kwargs)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if ret_rms is not None:
            # copy this here to make sure the new envs don't change the return stats where this comes from
            ret_rms = copy.copy(ret_rms)

        envs = VecNormalize(envs,
                            normalise_rew=normalise_rew, ret_rms=ret_rms,
                            gamma=0.99 if gamma is None else gamma,
                            cliprew=args.norm_rew_clip_param
                            )

    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset_mdp(self, index=None):
        obs = self.venv.reset_mdp(index=index)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset(self, index=None, task=None):
        if task is not None:
            assert isinstance(task, list)
        obs = self.venv.reset(index=index, task=task)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, list):  # raw + normalised
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.from_numpy(r).unsqueeze(dim=1).float().to(self.device) for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return obs, reward, done, info

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """

        if attr in ['_max_episode_steps', 'task_dim', 'belief_dim', 'num_states']:
            return self.unwrapped.get_env_attr(attr)

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
