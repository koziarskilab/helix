from typing import List

import numpy as np
import gin

from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionState


@gin.configurable()
class NoisyProxy(ProxyBase[ReactionState]):
    """Wraps a proxy and adds Gaussian noise to its output."""

    def __init__(self, base_proxy: ProxyBase[ReactionState], noise_std: float = 0.1, seed: int = 42):
        super().__init__()
        self.base_proxy = base_proxy
        self.noise_std = noise_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_proxy_output(self, states: List[ReactionState]) -> ProxyOutput:
        base_values = self.base_proxy.compute_proxy_output(states).value
        noise = self.rng.normal(loc=0, scale=self.noise_std, size=len(states))
        return ProxyOutput(value=base_values + noise, components=None)