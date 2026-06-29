from time import time

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer as _Trainer_
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from torch import cuda

from env import PROJECT_NAME, WANDB_ENTITY
from hydra.core.hydra_config import HydraConfig
from src.hydra_parsing import parse_hyperparams
from wandb.sdk.lib.runid import generate_id

class GradientMonitor_afterB(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        norms = []
        for p in pl_module.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        print(f"Gradient Norms after backward: {norms}")       
        
def _get_logger(cfg: DictConfig):
    name = f"seed{cfg.get('seed', '')}.{int(time())}"
    group_format = (
        "{dataset}.{causal_discovery}.{llm}.{rag}."
        "{model}.h{hidden_size}.lr{lr}"
    )
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.get("notes") is not None:
        group = f"{group}.{cfg.notes}"
    if cfg.trainer.logger == "wandb":
        logger = WandbLogger(
            project=PROJECT_NAME,
            entity=WANDB_ENTITY,
            log_model=True,
            id=generate_id(),
            save_dir=HydraConfig.get().runtime.output_dir,
            name=name,
            group=group,
        )
    else:
        raise ValueError(f"Unknown logger {cfg.trainer.logger}")
    return logger


class Trainer(_Trainer_):
    def __init__(self, cfg: DictConfig):
        callbacks = []
        checkpointing_enabled = cfg.trainer.get("enable_checkpointing", True)
        if cfg.trainer.get("monitor", None) is not None:
            if cfg.trainer.get("patience", None) is not None:
                # Delay early stopping until after propagator warmup
                min_delta = cfg.trainer.get("min_delta", 0.0)
                callbacks.append(
                    EarlyStopping(
                        monitor=cfg.trainer.monitor,
                        patience=cfg.trainer.patience,
                        mode=cfg.trainer.get("mode", "min"),
                        min_delta=min_delta,
                    )
                )
            if checkpointing_enabled:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath="checkpoints",
                        every_n_epochs=None,
                        monitor=cfg.trainer.monitor,
                        save_top_k=1,
                        mode=cfg.trainer.get("mode", "min"),
                        save_last=True,
                        save_weights_only=False,
                    )
                )
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            )
        )
        # callbacks.append(GradientMonitor_afterB())
        if cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"
        if cfg.trainer.get("logger") is not None:
            logger = _get_logger(cfg)
        else:
            logger = DummyLogger()
        trainer_kwargs = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["monitor", "patience", "logger", "mode", "min_delta"]
        }
        trainer_kwargs.setdefault("gradient_clip_val", 5.0)
        super().__init__(
            callbacks=callbacks,
            accelerator=accelerator,
            logger=logger,
            **trainer_kwargs,
        )
