"""
1-shot evaluation of the author’s pretrained ResNet-20 on CIFAR-10
Run this in your venv, no Lightning.
"""

import torch, torchvision as tv
from torch.utils.data import DataLoader
from torch.hub import load

# ------------------------------------------------------------
# 1.  Data pipeline THE MODEL EXPECTS
# ------------------------------------------------------------
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

test_tf = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean, std),
])

test_set = tv.datasets.CIFAR10("data/cifar10", train=False, transform=test_tf, download=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ------------------------------------------------------------
# 2.  Load PRE-TRAINED weights exactly as the author does
# ------------------------------------------------------------
model = load("chenyaofo/pytorch-cifar-models",
             "cifar10_resnet20",
             pretrained=True,
             verbose=False).cuda().eval()

# ------------------------------------------------------------
# 3.  Plain accuracy loop
# ------------------------------------------------------------
correct, total = 0, 0
with torch.no_grad(), torch.cuda.amp.autocast(False):
    for x, y in test_loader:
        logits = model(x.cuda(non_blocking=True))
        pred = logits.argmax(1)
        correct += (pred.cpu() == y).sum().item()
        total   += y.size(0)

print(f"Top-1 accuracy: {correct/total:.4f}")        # should print ~0.917