from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
from collections import OrderedDict
import numpy as np


def init_method_1(model):
    """Init uniform

    Args:
        model (_type_): _description_
    """
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_method_2(model):
    """Init normal

    Args:
        model (_type_): _description_
    """
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = ptu.build_mlp(input_size=self.ob_dim,
                               output_size=self.output_size,
                               n_layers=self.n_layers, size=self.size)
        self.f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                   output_size=self.output_size,
                                   n_layers=self.n_layers, size=self.size)

        # Init networks differently
        init_method_2(self.f_hat)
        init_method_1(self.f)

        # Optimizer
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        target = self.f(ob_no).detach()
        pred = self.f_hat(ob_no)

        # Return the L2 loss
        return ((pred - target)**2).sum(dim=1)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch

        # Fix issue with np arrays
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)
        # Get loss
        loss = self(ob_no).mean()

        # Take optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
