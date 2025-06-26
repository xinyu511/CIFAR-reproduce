import torch
import pytorch_lightning as pl
from lit_cifar import LitCIFAR
from data_module import CIFAR10DataModule

# Load pretrained model from Torch Hub
backbone = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
# Wrap in LightningModule
model = LitCIFAR(backbone)

# Load data
data = CIFAR10DataModule()

# Train on GPU if available
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1 if torch.cuda.is_available() else None)
trainer.fit(model, datamodule=data)