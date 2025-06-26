import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch, importlib
from lit_cifar import LitCIFAR

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))          # full config at runtime
    pl.seed_everything(cfg.seed, workers=True)

    # 1) datamodule -----------------------------------------------------------
    dm = instantiate(cfg.datamodule)

    # 2) model backbone from torch.hub ----------------------------------------
    backbone = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        cfg.model.name,
        pretrained=cfg.model.pretrained,
    )

    net = LitCIFAR(backbone, lr=cfg.model.lr)

    # 3) trainer ---------------------------------------------------------------
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.fit(net, datamodule=dm)

    # 4) optional final test ---------------------------------------------------
    if cfg.test_after_fit:
        trainer.test(net, datamodule=dm, ckpt_path="best" if trainer.checkpoint_callback else None)

if __name__ == "__main__":
    main()
