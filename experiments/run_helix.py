import argparse
from pathlib import Path

import gin
import pandas as pd
import torch

from rgfn.trainer.trainer import Trainer
from rgfn.bo.adapter import AcquisitionProxy, RGFN
from rgfn.utils.helpers import seed_everything

from molbo.bo import BOLoop, History

from experiments.utils import paths
from experiments.utils.factory import (
    RunConfig,
    make_acquisition,
    make_model,
    make_oracle,
)
from experiments.utils.metrics import (
    best_observed,
    n_observed,
    top_k_mean,
)


MODEL_NAMES = {"helix", "helix_bo"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--oracle", choices=["seh", "jnk3", "gsk"], default="seh")
    parser.add_argument("--hit", type=int, default=0)
    return parser.parse_args()


def derive_cfg_meta(cfg_path: str):
    """
    Map a cfg path to (experiment, model, variant)
    """
    cfg = Path(cfg_path)
    if cfg.parent.name == "experiments":
        return cfg.stem, "helix", None
    experiment = cfg.parent.name
    if cfg.stem in MODEL_NAMES:
        return experiment, cfg.stem, None
    return experiment, "helix", cfg.stem


def main():
    args = parse_args()
    seed_everything(args.hit)

    experiment, model, variant = derive_cfg_meta(args.cfg)
    name = paths.run_name(experiment, model, args.oracle, args.hit, variant=variant)

    gin.parse_config_files_and_bindings(
        [args.cfg],
        bindings=[
            f'hits_filepath = "{paths.hits_path(args.oracle)}"',
            f'load_starting_molecules.hit = {args.hit}',
            f'run_name = "{name}"',
        ],
    )

    paths.runs_dir(experiment, model, args.oracle, args.hit, variant=variant).mkdir(
        parents=True, exist_ok=True
    )

    # Construct RGFN components in float32, BO components require float64
    torch.set_default_dtype(torch.float32)
    proxy = AcquisitionProxy()
    trainer = Trainer()
    rgfn = RGFN()
    logger = trainer.logger
    torch.set_default_dtype(torch.float64)

    oracle = make_oracle(args.oracle)
    surrogate = make_model()
    acq_func = make_acquisition()
    config = RunConfig()

    ckpt_path = paths.checkpoint_path(experiment, model, args.oracle, args.hit, variant=variant)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    init = rgfn.sample_init(oracle=oracle, n_init=config.n_init)

    history = History(
        X_init=init.train_X,
        y_init=init.train_y,
        metrics=[best_observed, n_observed, top_k_mean(ks=[10, 100])],
        logger=logger,
        checkpoint_path=ckpt_path,
        checkpoint_freq=config.checkpoint_freq,
    )

    bo_loop = BOLoop(
        history=history,
        model=surrogate,
        acq_func=acq_func,
        oracle=oracle,
        acqf_optimizer=rgfn,
    )

    bo_loop.run(config.n_iters)

    # Write results
    results_path = paths.results_path(experiment, model, args.oracle, args.hit, variant=variant)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "smiles": history.X_all,
        "oracle_score": history.y_all.squeeze(-1).tolist(),
    }).to_csv(results_path, index=False)

    logger.log_to_file(gin.operative_config_str(), "operative_config")
    logger.log_to_file(gin.config_str(), "config")
    logger.close()


if __name__ == "__main__":
    main()
