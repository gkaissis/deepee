import os

os.environ["OMP_NUM_THREADS"] = "16"
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
import torchvision
from memory_profiler import memory_usage
import segmentation_models_pytorch as smp
from unet import UNet
from time import time

import warnings

warnings.filterwarnings("ignore")
import gc

gc.disable()

model = UNet(depth=3, padding=True)

model = convert_batchnorm_modules(model)

dataset = torchvision.datasets.FakeData(
    size=32,
    image_size=(1, 256, 256),
    num_classes=2,
    transform=torchvision.transforms.ToTensor(),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=0, pin_memory=False
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
privacy_engine = PrivacyEngine(
    model,
    sample_rate=1.0,
    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    secure_rng=False,
)
privacy_engine.attach(optimizer)
criterion = smp.utils.losses.DiceLoss()


def main():
    tick = time()
    model.train()
    optimizer.zero_grad()
    data, _ = next(iter(dataloader))
    output = model(data)
    loss = criterion(data, output)
    loss.backward()
    optimizer.step()
    tock = time()
    print(f"Took {tock-tick:.4f} seconds.")


if __name__ == "__main__":
    print("Opacus")
    print(max(memory_usage(main)))
