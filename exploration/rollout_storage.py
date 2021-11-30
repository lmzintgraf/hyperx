import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorage(object):
    def __init__(self, max_buffer_size, env_state_shape, belief_shape, action_shape):

        # count the number of datapoints seen so far (so we can do reservoir sampling)
        self.max_buffer_size = max_buffer_size

        # buffers for the data
        self.env_states = torch.zeros((self.max_buffer_size, *env_state_shape))
        self.beliefs = torch.zeros((self.max_buffer_size, *belief_shape))
        self.actions = torch.zeros((self.max_buffer_size, *action_shape))

        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

    def insert(self, env_states, beliefs, actions):

        # check where to insert data
        num_new = env_states.shape[0]
        if self.insert_idx + num_new > self.max_buffer_size:
            # keep track of how much we filled the buffer (for sampling from it)
            self.buffer_len = self.insert_idx
            # this will keep some entries at the end of the buffer without overwriting them,
            # but the buffer is large enough to make this negligible
            self.insert_idx = 0
        else:
            self.buffer_len = max(self.buffer_len, self.insert_idx + num_new)

        # insert new data
        self.env_states[self.insert_idx:self.insert_idx + num_new] = env_states
        if beliefs is not None:
            self.beliefs[self.insert_idx:self.insert_idx + num_new] = beliefs
        else:
            self.beliefs = None
        self.actions[self.insert_idx:self.insert_idx + num_new] = actions

        # count up current insert index
        self.insert_idx += num_new

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize):

        indices = np.random.choice(range(self.buffer_len), batchsize)

        if self.beliefs is not None:
            return self.env_states[indices], self.beliefs[indices], self.actions[indices]
        else:
            return self.env_states[indices], None, self.actions[indices]
