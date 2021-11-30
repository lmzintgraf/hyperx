import torch

from exploration.rnd.models import RNDPriorNetwork, RNDPredictorNetwork
from utils.helpers import RunningMeanStd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNDRewardBonus:
    def __init__(self, args, logger, dim_inputs, rollout_storage):

        self.args = args
        self.logger = logger
        self.dim_input = dim_inputs
        self.rollout_storage = rollout_storage

        # initialise the random prior network (stays fixed)
        self.rnd_prior_net = RNDPriorNetwork(
            dim_inputs=dim_inputs,
            layers=self.args.rnd_prior_net_layers,
            dim_output=self.args.rnd_output_dim,
            weight_scale=self.args.rnd_init_weight_scale
        ).to(device)
        # can be set to eval mode since we don't need gradients
        self.rnd_prior_net.eval()

        # initialise the predictor network
        self.rnd_predictor_net = RNDPredictorNetwork(
            input_size=dim_inputs,
            layers=self.args.rnd_predictor_net_layers,
            dim_output=self.args.rnd_output_dim,
        ).to(device)
        # optimiser for the predictor net
        self.rnd_optimiser = torch.optim.Adam(self.rnd_predictor_net.parameters(), lr=self.args.rnd_lr)

        # normalisation parameters
        self.input_rms = [RunningMeanStd(shape=d) for d in dim_inputs]
        self.epsilon = 1e-8

        self.already_updated = False

    def _normalise_input(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in range(len(inputs)):
            inputs[i][..., self.input_rms[i].var != 0] /= torch.sqrt(self.input_rms[i].var[self.input_rms[i].var != 0] + self.epsilon)
        return inputs

    def _update_normalisation(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in range(len(inputs)):
            # update the normalisation params for the inputs
            self.input_rms[i].update(inputs[i])

    def reward(self, inputs, update_normalisation=False):

        if update_normalisation:
            self._update_normalisation(inputs)

        if self.args.rnd_norm_inputs:
            inputs = self._normalise_input(inputs)

        # get outputs from the RND prior and predictor
        output_prior = self.rnd_prior_net(inputs)
        output_predictor = self.rnd_predictor_net(inputs)

        # the difference is the reward bonus (average across output dimensions)
        rew_bonus = (output_prior - output_predictor).pow(2).mean(dim=-1).unsqueeze(-1)

        return rew_bonus

    def update(self, inputs):

        self.already_updated = True

        if self.args.rnd_norm_inputs:
            inputs = self._normalise_input(inputs)

        # get outputs from the RND prior and predictor
        output_prior = self.rnd_prior_net(inputs)
        output_predictor = self.rnd_predictor_net(inputs)

        # compute the MSE between the RND prior and predictor
        loss = (output_prior - output_predictor).pow(2).mean(dim=1).mean(dim=0)

        # update
        self.rnd_optimiser.zero_grad()
        loss.backward()
        self.rnd_optimiser.step()

        return loss
