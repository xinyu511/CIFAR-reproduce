import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Optional


class LitCIFAR(pl.LightningModule):
    """
    Works for both CIFAR-10 (10 classes) and CIFAR-100 (100 classes).

    Args
    ----
    model : nn.Module            backbone from torch.hub
    num_classes : int | None     if None, try to infer from model.fc.out_features
    lr : float                   base learning-rate
    """
    def __init__(self, model, num_classes: Optional[int] = None, lr: float = 0.1):
        super().__init__()
        self.model = model
        self.lr = lr

        # ─── figure out #classes ──────────────────────────────────────────────
        if num_classes is None:
            # common patterns: ResNet, VGG, MobileNet, RepVGG …
            for attr in ("fc", "classifier", "head", "linear"):
                layer = getattr(model, attr, None)
                if isinstance(layer, torch.nn.Linear):
                    num_classes = layer.out_features
                    break
            else:            # fallback
                raise ValueError("num_classes is None and could not be inferred")

        self.save_hyperparameters("lr", "num_classes") 

        # ─── Metrics ──────────────────────────────────────────────────────────
        self.train_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.train_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=min(5, num_classes))
        self.val_top1   = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_top5   = Accuracy(task="multiclass", num_classes=num_classes, top_k=min(5, num_classes))

    # ─── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    # ─── Train loop ───────────────────────────────────────────────────────────
    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc1", self.train_top1(logits, y),
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("train_acc5", self.train_top5(logits, y),
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=len(y))
        return loss

    # ─── Validation loop ─────────────────────────────────────────────────────
    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

        self.log("val_acc1", self.val_top1(logits, y), prog_bar=True)
        self.log("val_acc5", self.val_top5(logits, y))

    # ─── Optimizer & LR-scheduler ────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=5e-4, nesterov=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=0.0)
        return {"optimizer": opt, "lr_scheduler": sch}