from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from torch import distributions


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20, 2],
                 action_range=[-1, 1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers,
                                           size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(
            np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        distribution = self(obs)

        if sample:
            action = distribution.sample()
        else:
            action = distribution.mean

        # Clip grads to action range
        acr_min, acr_max = self.action_range

        action = torch.clip(action, min=acr_min, max=acr_max)

        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!

    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        self.logstd.tanh()
        super()(observation)

        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file

        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            # Mean
            batch_mu = self.mean_net(observation)

            # Clip log
            log_std_min, log_std_max = self.log_std_bounds[0], self.log_std_bounds[1]

            clipped_log_std = torch.exp(
                self.log_std.clamp(min=log_std_min, max=log_std_max))
            # Create squashed normal
            action_distribution = sac_utils.SquashedNormal(
                batch_mu, clipped_log_std)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer

        # Get the action, sample, and logprob
        act_dist = self(obs)

        # Keep computational graph alive
        act_s = act_dist.rsample()

        # Loglikelihood
        act_s_logl = act_dist.log_prob(act_s)

        # Get q values
        q_1, q_2 = critic(obs, act_s)
        q = torch.minimum(q_1, q_2)

        # Calculate the actor loss
        actor_loss = self.alpha.detach() * act_s_logl - q
        actor_loss = actor_loss.mean()

        # Normal backprob stuff for actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Calculate the alpha loss
        alpha_loss = - self.alpha * (act_s_logl + self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()

        # Optimize actor critic
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha
