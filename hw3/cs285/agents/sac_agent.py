from collections import OrderedDict

import torch

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import cs285.infrastructure.sac_utils as sau


class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO:
        # 1. Compute the target Q value.
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic

        # Compute targets
        with torch.no_grad():
            action_dist = self.actor(next_ob_no)
            act_t1 = action_dist.rsample()
            act_t1_logprobs = action_dist.log_prob(act_t1)

            # Compute q
            n_q1, n_q2 = self.critic_target(next_ob_no, act_t1)
            n_q = torch.minimum(n_q1, n_q2)

            # target
            target = re_n + self.gamma * \
                (1 - terminal_n) * (n_q - self.actor.alpha * act_t1_logprobs)

        q1, q2 = self.critic(ob_no, ac_na)

        critic_loss = self.critic.loss(
            q1, target) + self.critic.loss(q2, target)

        # Optimize and take step
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # for agent_params['num_critic_updates_per_agent_update'] steps,

        # Accumulate loss
        critic_loss = 0.
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no=ptu.from_numpy(ob_no), ac_na=ptu.from_numpy(ac_na), re_n=ptu.from_numpy(
                re_n), next_ob_no=ptu.from_numpy(next_ob_no), terminal_n=ptu.from_numpy(terminal_n))
        # average
        critic_loss = critic_loss/(i+1)

        # Only at specific freq update target critic
        if self.training_step % self.critic_target_update_frequency == 0:
            sau.soft_update_params(
                self.critic, self.critic_target, self.critic_tau)

        # Only update once in a while

        # WARN: Unsure if logging can handle empty values
        actor_l = 0.
        alpha_l = 0.
        alpha = self.actor.alpha
        if self.training_step % self.actor_update_frequency == 0:
            for i in range(self.agent_params['num_actor_updates_per_agent_update']):
                act_l_t, alp_l_t, alpha = self.actor.update(
                    ptu.from_numpy(ob_no), self.critic)
                actor_l += act_l_t
                alpha_l += alp_l_t

            # mean
            actor_l /= (i+1)
            alpha_l /= (i+1)

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_l
        loss['Alpha_Loss'] = alpha_l
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
