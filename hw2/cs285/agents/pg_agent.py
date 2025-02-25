import numpy as np
# Weird it's not mentioned in comments
from cs285.infrastructure.utils import normalize
from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from math import isclose


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):
        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO update the PG actor/policy using the given batch of data
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        # Compute the Q values
        q_vals = self.calculate_q_vals(rewards_list=rewards_list)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_vals, terminals)

        # Update the agent policy
        train_log = self.actor.update(
            observations, actions, advantages, q_vals)

        return train_log

    def calculate_q_vals(self, rewards_list):
        """
            Monte Carlo estimation of the Q function.
        """

        # TODO return the estimated qvals based on the given rewards, using
        # either the full trajectory-based estimator or the reward-to-go
        # estimator

        # Note: rewards_list is a list of lists of rewards with the inner list
        # being the list of rewards for a single trajectory.

        # HINT: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these).

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should be a 2D numpy array where the first
        # dimension corresponds to trajectories and the second corresponds
        # to timesteps

        if not self.reward_to_go:
            q_values = map(self._discounted_return, rewards_list)

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values = map(self._discounted_cumsum, rewards_list)

        # concat arrays
        return np.concatenate(list(q_values))

    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):
        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            # ensure that the value predictions and q_values have the same dimensionality
            # to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            # TODO values were trained with standardized q_values, so ensure
            # that the predictions have the same mean and standard deviation as
            # the current batch of q_values

            # https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

            # define values to use
            q_mu = np.mean(q_values)
            q_std = np.std(q_values)
            v_mu = np.mean(values_unnormalized)
            v_std = np.std(values_unnormalized)
            # Calculate new set
            values = values_unnormalized*(q_std/v_std)
            values = values + q_mu-v_mu*(q_std/v_std)
            # Assert correct
            assert isclose(np.mean(values), q_mu, abs_tol=1e-4)
            assert isclose(np.std(values), q_std, abs_tol=1e-4)

            if self.gae_lambda is not None:
                # append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                # combine rews_list into a single array
                rews = np.concatenate(rews_list)

                # create empty numpy array to populate with GAE advantage
                # estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO recursively compute advantage estimates starting from
                    # timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i]
                    # is 1 if the state is the last in its trajectory, and
                    # 0 otherwise.

                    # Edgecase where we don't have the values[i+1]
                    if terminals[i] == 1:
                        advantages[i] = rews[i]-values[i]
                        continue
                    # The ordinary case as described by (21) in the problem formulation
                    delta_t = (rews[i] + self.gamma * values[i+1] - values[i])
                    advantages[i] = delta_t + self.gamma * \
                        self.gae_lambda*advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                # TODO compute advantage estimates using q_values, and values as baselines
                advantages = q_values-values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(
                advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        list_of_discounted_returns = np.ones(len(rewards))

        # each element of the sum
        _sum = sum([np.power(self.gamma, i)*rewards[i]
                   for i in range(len(rewards))])
        # Add sum to list of discounted returns
        list_of_discounted_returns *= _sum

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        """
        list_of_discounted_cumsums = []
        T = len(rewards)
        for i in range(T):
            _idx = np.arange(i, T)
            _sum_step = np.sum(
                np.power(self.gamma, _idx-i) * rewards[_idx])
            list_of_discounted_cumsums.append(_sum_step)

        return np.array(list_of_discounted_cumsums)
