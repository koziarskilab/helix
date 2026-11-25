import argparse
from pathlib import Path

import gin
from gin_config.utils import get_time_stamp

from rgfn.trainer.trainer import Trainer
from rgfn.utils.helpers import seed_everything


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

    trainer = Trainer()
    trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    trainer.logger.log_to_file(gin.config_str(), "config")
    trainer.train()
    trainer.logger.close()


if __name__ == "__main__":
    main()