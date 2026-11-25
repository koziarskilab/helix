import gin
import torch
from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateEarlyTerminal
from rgfn.shared.policies.uniform_policy import UniformPolicy
from rgfn.shared.samplers.random_sampler import RandomSampler

from molbo.acqf_optimizer.base import AcqfOptimizer, Initialization, OptimizationResult


@gin.configurable()
class AcquisitionProxy(ProxyBase):
    """Wraps a molbo acquisition function as an RGFN proxy."""

    def __init__(self):
        super().__init__()
        self.acq_func = None

    def update(self, acq_func):
        self.acq_func = acq_func

    def compute_proxy_output(self, states):
        if self.acq_func is None:
            raise RuntimeError("acq_func not set - call update() before training.")

        valid_mask = [not isinstance(s, ReactionStateEarlyTerminal) for s in states]
        valid_states = [s for s, v in zip(states, valid_mask) if v]

        if not valid_states:
            return ProxyOutput(value=torch.zeros(len(states), dtype=torch.float32), components=None)

        smiles = [s.molecule.smiles for s in valid_states]

        with torch.no_grad():
            acq_values = self.acq_func(smiles).float()  # back to float32 for RGFN

        if torch.isnan(acq_values).any():
            print("Warning: NaN acquisition values, replacing with zeros")
            acq_values = torch.nan_to_num(acq_values, nan=0.0)

        result = torch.zeros(len(states), dtype=torch.float32)
        result[torch.tensor(valid_mask)] = acq_values
        return ProxyOutput(value=result, components=None)

    @property
    def is_non_negative(self):
        return True

    @property
    def higher_is_better(self):
        return True

    @property
    def n_proxy_calls(self) -> int:
        """Dummy property for compatibility with Trainer oracle budget check."""
        return 0

@gin.configurable()
class RGFN(AcqfOptimizer):

    def __init__(
        self,
        trainer,
        proxy: AcquisitionProxy,
        M: int = 1000,
        q: int = 500,
        n_train_iterations: int = 200,
    ):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.trainer = trainer
        torch.set_default_dtype(prev)

        self.proxy = proxy
        self.M = M
        self.q = q
        self.n_train_iterations = n_train_iterations

    def _optimize(self, acq_func, candidates=None, observed_indices=None):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        self.proxy.update(acq_func)
        self.trainer.n_iterations = self.trainer.start_iteration + self.n_train_iterations
        self.trainer.train()
        self.trainer.start_iteration = self.trainer.n_iterations

        seen_smiles = set()
        terminal_states = []
        while len(terminal_states) < self.M:
            trajectories = self.trainer.train_forward_sampler.sample_trajectories_batch(
                n_total_trajectories=self.M - len(terminal_states),
                batch_size=self.M - len(terminal_states),
            )
            for s in trajectories.get_last_states_flat():
                if not isinstance(s, ReactionStateEarlyTerminal) and s.molecule.smiles not in seen_smiles:
                    seen_smiles.add(s.molecule.smiles)
                    terminal_states.append(s)
                if len(terminal_states) == self.M:
                    break

        smiles = [s.molecule.smiles for s in terminal_states]
        torch.set_default_dtype(prev_dtype)  # restore before acq_func evaluation

        with torch.no_grad():
            acq_values = acq_func(smiles)  # (M,)

        top_q = acq_values.topk(min(self.q, len(smiles)))
        selected_smiles = [smiles[i] for i in top_q.indices.tolist()]

        return OptimizationResult(new_X=selected_smiles, acq_val=top_q.values)

    def sample_init(self, oracle, n_init: int) -> Initialization:
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        uniform_sampler = RandomSampler(
            env=self.trainer.train_forward_sampler.env,
            policy=UniformPolicy(),
            reward=None,
        )

        seen_smiles = set()
        sampled_states = []
        while len(sampled_states) < n_init:
            trajectories = uniform_sampler.sample_trajectories_batch(
                n_total_trajectories=n_init - len(sampled_states),
                batch_size=n_init - len(sampled_states),
            )
            for state in trajectories.get_last_states_flat():
                if not isinstance(state, ReactionStateEarlyTerminal) and state.molecule.smiles not in seen_smiles:
                    seen_smiles.add(state.molecule.smiles)
                    sampled_states.append(state)

        torch.set_default_dtype(prev_dtype)

        smiles = [s.molecule.smiles for s in sampled_states]
        train_y = oracle(smiles)  # (n_init, 1) float64
        return Initialization(train_X=smiles, train_y=train_y)
