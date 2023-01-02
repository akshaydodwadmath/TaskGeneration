import torch


class Rolls(object):

    def __init__(self, action, proba, multiplicity, depth, device=None):
        self.device = torch.device(
            device if torch.cuda.is_available() and 'cuda' in device else "cpu")
        self.device_name = device
        self.successor = {}
        # The action that this node in the tree corresponds to
        self.action = action  # -> what was the sample taken here
        self.proba = proba  # -> Variable containing the log proba of
        # taking this action
        self.multi_of_this = multiplicity  # -> How many times was this
        # prefix (until this point of
        # the sequence) seen
        self.depth = depth  # -> How far along are we in the sequence

        # Has no successor to this sample
        self.is_final = True

        # This is the reward that would be obtained by doing this prefix once.
        # This is only to use for bookkeeping.
        self.own_reward = 0
        # The one to use to compute gradients is the following.

        # This contains `self.own_reward * self.multi_of_this` + sum of all the
        # dependents self.dep_reward
        self.dep_reward = 0

    def expand_samples(self, trajectory, end_multiplicity, end_proba):
        """
        The assumption here is that all but the last steps of the trajectory
        have already been created.
        """
        assert (len(trajectory) > 0)

        pick = trajectory[0]
        if pick in self.successor:
            self.successor[pick].expand_samples(trajectory[1:],
                                                end_multiplicity,
                                                end_proba)
        else:
            # We add a successor so we are necessarily not final anymore
            self.is_final = False
            # We don't expand the samples by several steps at a time so verify
            # that we are done
            assert (len(trajectory) == 1)
            self.successor[pick] = Rolls(pick, end_proba,
                                         end_multiplicity,
                                         self.depth + 1, self.device_name)

    def yield_final_trajectories(self):
        """
        Yields 3-tuples:
        -> Trajectory
        -> Multiplicity of this trajectory
        -> Proba of this trajectory
        -> Final reward of this trajectory
        """
        if self.is_final:
            yield [], self.multi_of_this, self.proba, self.own_reward
        else:
            for key, succ in self.successor.items():
                for final_traj, multi, proba_suffix, reward \
                        in succ.yield_final_trajectories():
                    yield ([key] + final_traj,
                           multi,
                           self.proba * proba_suffix,
                           reward)

    def yield_var_and_grad(self):
        """
        Yields 2-tuples:
        -> Proba: Variable correponding to the proba of this last choice
        -> Grad: Gradients for each of those variables
        """
        for succ in self.successor.values():
            for var, grad in succ.yield_var_and_grad():
                yield var, grad
        yield self.proba, self.reinforce_gradient()

    def assign_rewards(self, env, trace, task_id):
        """
        Using the `reward_assigner` scorer, go depth first to assign the
        reward at each timestep, and then collect back all the "depending
        rewards"
        """

        # Assign their own score to each of the successorEnv
        for next_step, succ in self.successor.items():
            new_trace = trace + [next_step]
            succ.assign_rewards(env, new_trace, task_id)

        # If this is a final node, there is no successor, so I can already
        # compute the dep-reward.
        if self.successor == {}:
            _, rew, _, _ = env.multistep(trace, task_id)
            self.dep_reward = self.multi_of_this * rew
            # self.dep_reward = self.multi_of_this * self.own_reward
        else:
            # On the other hand, all my child nodes have already computed their
            # dep_reward so I can collect them to compute mine
            self.dep_reward = self.multi_of_this * self.own_reward
            for succ in self.successor.values():
                self.dep_reward += succ.dep_reward

    def reinforce_gradient(self):
        """
        At each decision, compute a reinforce gradient estimate to the
        parameter of the probability that was sampled from.
        """
        if self.depth == -1:
            return None
        else:
            # We haven't put in a baseline so just ignore this
            baselined_reward = self.dep_reward
            grad_value = baselined_reward / (1e-6 + self.proba.data)

            # We return a negative here because we want to maximize the rewards
            # And the pytorch optimizers try to minimize them, so this put them
            # in agreement
            return -grad_value
