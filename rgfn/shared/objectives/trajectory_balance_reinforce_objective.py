from typing import Iterator

import gin
import torch
from torch import nn
from torch.nn import Parameter

from rgfn.api.env_base import EnvBase
from rgfn.api.objective_base import ObjectiveBase, ObjectiveOutput
from rgfn.api.policy_base import PolicyBase
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.api.trajectory_filter_base import (
    IdentityTrajectoryFilter,
    TrajectoryFilterBase,
)
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
class TrajectoryBalanceReinforceObjective(ObjectiveBase[TState, TActionSpace, TAction]):
    """
    Trajectory balance objective for GFN from the paper "Trajectory balance: Improved credit assignment in GFlowNets"
    (https://arxiv.org/abs/2201.13259)

    Attributes:
        forward_policy: a policy that estimates the probabilities of actions taken in the forward direction.
        backward_policy: a policy that estimates the probabilities of actions taken in the backward direction.
        logZ: a learnable parameter that represents the log partition function.
    """

    def __init__(
        self,
        forward_policy: PolicyBase[TState, TActionSpace, TAction],
        backward_policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        trajectory_filter: TrajectoryFilterBase[TState, TActionSpace, TAction] | None = None,
        detach_backward: bool = True,
        entropy_multiplier: float = 1.0,
        z_dim: int = 1,
    ):
        super().__init__(forward_policy=forward_policy, backward_policy=backward_policy)
        self.logZ = nn.Parameter(torch.ones(z_dim) * 150.0 / 64)
        self.trajectory_filter = trajectory_filter or IdentityTrajectoryFilter()
        self.env = env
        self.detach_backward = detach_backward
        self.entropy_multiplier = entropy_multiplier

    def compute_objective_output(
        self, trajectories_container: TrajectoriesContainer[TState, TActionSpace, TAction]
    ) -> ObjectiveOutput:
        """
        Compute the objective output on a batch of trajectories.

        Args:
            trajectories_container: the batch of trajectories obtained in the sampling process. It contains the states, actions,
                action spaces in forward and backward directions, and rewards. Other important quantities (e.g. log
                probabilities of taking actions in forward and backward directions) should be assigned in this method
                using appropriate methods (e.g. assign_log_probs).

        Returns:
            The output of the objective function, containing the loss and possibly some metrics.
        """
        trajectories_container = self.trajectory_filter(trajectories_container)
        trajectories = trajectories_container.get_all_trajectories()
        self.assign_log_probs(trajectories)
        trajectories_container.objective_trajectories = trajectories

        forward_mask = trajectories_container.get_forward_mask_for_all()
        replay_mask = trajectories_container.get_replay_mask_for_all()
        reinforce_mask = trajectories_container.get_reinforce_mask_for_all()

        forward_trajectories = trajectories.masked_select(forward_mask | replay_mask)
        forward_log_prob = forward_trajectories.get_forward_log_probs_flat()  # [n_actions]
        backward_log_prob = forward_trajectories.get_backward_log_probs_flat()  # [n_actions]
        if self.detach_backward:
            backward_log_prob = backward_log_prob.detach()
        log_reward = forward_trajectories.get_reward_outputs().log_reward  # [n_trajectories]
        index = forward_trajectories.get_index_flat().to(self.device)  # [n_actions]

        tb_loss = torch.scatter_add(
            input=self.logZ.sum() - log_reward,
            index=index,
            src=forward_log_prob - backward_log_prob,
            dim=0,
        )  # [n_trajectories]
        tb_loss = tb_loss.pow(2).mean()

        reinforce_trajectories = trajectories.masked_select(forward_mask | reinforce_mask)
        backward_log_prob = reinforce_trajectories.get_backward_log_probs_flat()
        backward_entropy = reinforce_trajectories.get_backward_entropies_flat()
        index = reinforce_trajectories.get_index_flat().to(self.device)
        input_tensor = torch.zeros(
            len(reinforce_trajectories), device=self.device, dtype=torch.float
        )
        reinforce_backward_log_prob = torch.scatter_add(
            input=input_tensor, index=index, src=backward_log_prob, dim=0
        )
        reinforce_backward_entropy = torch.scatter_add(
            input=input_tensor, index=index, src=backward_entropy, dim=0
        )
        source_states = reinforce_trajectories.get_source_states_flat()
        source_states_mask = self.env.get_valid_source_mask(source_states)
        reinforce_rewards = torch.tensor(source_states_mask).long().to(self.device) * 2 - 1

        reinforce_loss = \
            -(reinforce_backward_log_prob * reinforce_rewards).mean() \
            - self.entropy_multiplier * reinforce_backward_entropy.mean()

        loss = tb_loss + reinforce_loss
        return ObjectiveOutput(loss=loss, metrics={
            "logZ": self.logZ.sum().item(),
            "tb_loss": tb_loss.item(),
            "reinforce_loss": reinforce_loss.item(),
        })

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Get the parameters of the objective function.

        Args:
            recurse: whether to recursively get the parameters of the submodules.

        Returns:
            An iterator over the parameters.
        """
        yield from super().parameters(recurse)
        yield self.logZ
