from dataclasses import dataclass

import gin

from rgfn.bo.oracle import SEHOracle, JNK3Oracle, GSK3bOracle

from molbo.acquisition import EIAcquisition, PosteriorMeanAcquisition, UCBAcquisition
from molbo.model import TanimotoGPModel
from molbo.model.featurizer import MorganFingerprintFeaturizer


@gin.configurable
@dataclass
class RunConfig:
    """Loop-level orchestration parameters."""
    n_init: int = 50
    n_iters: int = 20
    checkpoint_freq: int = 1


@gin.configurable
def make_acquisition(
    acq: str = "ucb",
    beta: float = 1.0,
):
    """
    Args:
        kind: "ucb", "ei", or "pm"
        beta: UCB exploration parameter (ignored unless kind="ucb")
    """
    if acq == "ucb":
        return UCBAcquisition(beta=beta)
    elif acq == "ei":
        return EIAcquisition()
    elif acq == "pm":
        return PosteriorMeanAcquisition()
    else:
        raise ValueError(f"Unknown acquisition kind: {acq!r}")


@gin.configurable
def make_model():
    """Factory for the Tanimoto GP surrogate."""
    return TanimotoGPModel(featurizer=MorganFingerprintFeaturizer())


@gin.configurable
def make_oracle(kind: str = "seh", noise_std: float = 0.0):
    """Factory for the oracle"""
    ORACLE_MAP = {
        "seh": SEHOracle,
        "jnk3": JNK3Oracle,
        "gsk": GSK3bOracle,
    }
    return ORACLE_MAP[kind](noise_std=noise_std)

@gin.configurable
def make_logger(name: str, project: str = "helix", mode: str = "online"):
    from molbo.utils.logger import WandBLogger
    group = "/".join(name.split("/")[:-1])  # e.g. "helix/seh"
    return WandBLogger(project=project, name=name, group=group, mode=mode)
