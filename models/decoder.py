import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 latent_embed_dim=None,
                 input_prev_state=True,
                 input_action=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.input_prev_state = input_prev_state
        self.input_action = input_action

        # get state as input and predict reward prob
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        if latent_embed_dim is not None:
            self.latent_encoder = utl.FeatureExtractor(latent_dim, latent_embed_dim, F.relu)
            latent_dim = latent_embed_dim
        else:
            self.latent_encoder = None
        curr_input_dim = latent_dim + state_embed_dim
        if input_prev_state:
            curr_input_dim += state_embed_dim
        if input_action:
            curr_input_dim += action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):

        # we do the action-normalisation (the the env bounds) here
        if actions is not None:
            actions = utl.squash_action(actions, self.args)

        if self.latent_encoder is not None:
            latent_state = self.latent_encoder(latent_state)

        hns = self.state_encoder(next_state)
        h = torch.cat((latent_state, hns), dim=-1)
        if self.input_action:
            ha = self.action_encoder(actions)
            h = torch.cat((h, ha), dim=-1)
        if self.input_prev_state:
            hps = self.state_encoder(prev_state)
            h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 ):
        super(TaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):

        h = latent_state

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)
