import os

os.environ["OMP_NUM_THREADS"] = "16"
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
import torchvision
from memory_profiler import profile
import segmentation_models_pytorch as smp

import warnings

warnings.filterwarnings("ignore")
import gc

gc.disable()

model = torchvision.models.vgg11_bn()
model.classifier = torch.nn.Linear(512 * 7 * 7, 1)

model = convert_batchnorm_modules(model)

dataset = torchvision.datasets.FakeData(
    size=32,
    image_size=(3, 256, 256),
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
criterion = torch.nn.BCEWithLogitsLoss()


@profile
def main():
    model.train()
    optimizer.zero_grad()
    data, label = next(iter(dataloader))
    output = model(data)
    loss = criterion(label[..., None].to(float), output)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    print("Opacus")
    main()
