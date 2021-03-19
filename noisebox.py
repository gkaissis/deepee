import torch
from torchvision.models import resnet18
from deepee.noisebox import NoiseBox

box = NoiseBox(resnet18())
with open("noise.bin", "wb") as f:
    f.write(box.generate_noise_supply(30, 200, 1.0, 1.0))
