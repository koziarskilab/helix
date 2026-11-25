import argparse
from pathlib import Path

import gin
import torch
from gin_config.utils import get_time_stamp

from rgfn.trainer.trainer import Trainer
from rgfn.bo.adapter import AcquisitionProxy, RGFN
from rgfn.utils.helpers import seed_everything

from molbo.bo import BOLoop, History

from experiments.utils.factory import RunConfig, make_acquisition, make_model, make_oracle
from experiments.utils.metrics import best_observed, n_observed, top_k_mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    gin.parse_config_files_and_bindings([args.cfg], bindings=[])
    base_name = gin.query_parameter("%run_name")
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [args.cfg],
        bindings=[f'run_name="{base_name}/{get_time_stamp()}"']
    )

    run_name = gin.query_parameter("%run_name")
    user_root_dir = gin.query_parameter("%user_root_dir")
    run_dir = Path(user_root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path("checkpoints") / run_name / "history.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float32)
    proxy = AcquisitionProxy()
    trainer = Trainer()
    rgfn = RGFN()
    logger = trainer.logger
    torch.set_default_dtype(torch.float64)

    oracle = make_oracle()
    model = make_model()
    acq_func = make_acquisition()
    config = RunConfig()

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
        model=model,
        acq_func=acq_func,
        oracle=oracle,
        acqf_optimizer=rgfn,
    )

    bo_loop.run(config.n_iters)

    logger.log_to_file(gin.operative_config_str(), "operative_config")
    logger.log_to_file(gin.config_str(), "config")
    logger.close()


if __name__ == "__main__":
    main()