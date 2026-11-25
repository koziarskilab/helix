from typing import List

import torch

from molbo.oracle.molecular import MolecularOracle
from rgfn.gfns.reaction_gfn.proxies.seh_proxy import SEHProxyWrapper


class SEHOracle(MolecularOracle):
    def __init__(self, batch_size: int = 64, noise_std: float = 0.0):
        super().__init__(noise_std=noise_std)
        self.proxy = SEHProxyWrapper(batch_size=batch_size)

    def _evaluate(self, smiles: List[str]) -> torch.Tensor:
        scores = self.proxy.compute_scores(smiles)
        return torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)  # (N, 1)


class JNK3Oracle(MolecularOracle):
    def __init__(self, noise_std: float = 0.0):
        super().__init__(noise_std=noise_std)
        from tdc import Oracle
        self._oracle = Oracle(name="JNK3")

    def _evaluate(self, smiles: List[str]) -> torch.Tensor:
        scores = self._oracle(smiles)
        return torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)


class GSK3bOracle(MolecularOracle):
    def __init__(self, noise_std: float = 0.0):
        super().__init__(noise_std=noise_std)
        from tdc import Oracle
        self._oracle = Oracle(name="GSK3b")

    def _evaluate(self, smiles: List[str]) -> torch.Tensor:
        scores = self._oracle(smiles)
        return torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)
