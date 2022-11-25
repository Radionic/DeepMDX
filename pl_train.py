import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from loguru import logger
from icecream import ic

from src.datamodule import DeepMDXDataModule
from src.pl_deepmdx import DeepMDX


@hydra.main(version_base=None, config_path="src/config", config_name="debug")
def main(cfg: DictConfig) -> None:
    ic(dict(cfg))
    pl.seed_everything(cfg.seed)
    
    logger = instantiate(cfg.logger)
    
    model = DeepMDX(**cfg.model)
    if cfg['ckpt_path'] is not None:
        model.load_from_checkpoint(cfg['ckpt_path'])
    dm = DeepMDXDataModule(**cfg.datamodule)
    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()