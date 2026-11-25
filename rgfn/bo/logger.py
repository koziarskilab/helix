import gin
from molbo.bo.logger import Logger as MolboLogger
from rgfn.trainer.logger.wandb_logger import WandbLogger


@gin.configurable
class BOLogger(WandbLogger, MolboLogger):
    """RGFN's WandbLogger + molbo's Logger ABC. 
    
    BO metrics logged via History call log(data) -> log_metrics(data, prefix='bo').
    Trainer metrics logged via log_metrics(..., prefix='train') as before.
    """

    def log(self, data: dict) -> None:
        self.log_metrics(data, prefix="bo")
